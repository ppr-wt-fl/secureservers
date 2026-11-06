"""Plaintext EPPRFL against S0 and S1 over HTTP, on the same client updates.

Each round CLIENTS_PER_ROUND iid clients fetch the model and their seed from S0 (/model), train from it
(fedavg.local_train) and upload their shares to S1 (/upload), as the extension does. The same updates go to the oracle
(oracle/, copied from poisoning-eval, or ORACLE), and the servers' next model is compared with the oracle's.
The next round starts from the servers' model. The oracle keeps its own state, so a drift between the two shows up
round over round. The first ATTACKERS clients of each round send minus the mean of their own honest updates instead
(poisoning-eval's SignFlippingAttack); ATTACKERS=0 for honest rounds only. Exits 1 if any round's model or benign/clipped counts
differ from the oracle's.

Per round, to RESULTS/n{CLIENTS_PER_ROUND}-a{ATTACKERS}.jsonl: the oracle's and the servers' benign and clipped
counts, the model difference, test metrics of both models, the plaintext aggregation time (the oracle, in-process)
against the servers' (S0's /stats: compile, party 0's MPC run, glue, /arrived to the next round), MP-SPDZ's traffic
and rounds between S0 and S1, and the client traffic the protocol fixes (model down, shares up) against plaintext FL.

`python compare.py export DIR` writes fedavg's model as model.json + shard for S0's MODEL_DIR (in the weight order
get_flat uses; S0 only needs the trainable names in order).
Run with Docker, both attacker settings from fresh server state: mltest/run.sh
"""
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.append(os.environ.get('ORACLE', str(HERE / 'oracle')))
import party  # noqa: E402
import shares  # noqa: E402
from attacks.allenzhuByzantineResilientNonConvex2020 import SignFlippingAttack  # noqa: E402
from epprfl import EPPRFL  # noqa: E402
from fedavg import N_CLIENTS, evaluate, fmt, get_flat, local_train, model, partitions, rng  # noqa: E402

S0_URL = os.environ.get('S0_URL', 'http://localhost:8000')
S1_URL = os.environ.get('S1_URL', 'http://localhost:8001')
# robustness-evaluation's setup: all 200 clients every round for 50 rounds. The MPC cost grows with the clipped
# (honest) clients, ~1.2 GB per party each
CLIENTS_PER_ROUND = int(os.environ.get('CLIENTS_PER_ROUND', N_CLIENTS))
N_ROUNDS = int(os.environ.get('N_ROUNDS', 50))
ATTACKERS = int(os.environ.get('ATTACKERS', 0))
# uploads start only with this much of S1's window left, so all of them land in the same window (200 uploads of 8 MB)
UPLOAD_MARGIN_S = int(os.environ.get('UPLOAD_MARGIN_S', 45))
TOLERANCE = 1e-3
RESULTS = Path(os.environ.get('RESULTS', HERE / 'results'))


def export(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    linears = [name for name, _ in model.named_parameters() if name.endswith('.weight')]
    weights = [{'name': f'{w}/{kind}', 'shape': list(p.shape), 'dtype': 'float32'}
               for w in linears for kind, p in [('kernel', model.get_parameter(w)),
                                                ('bias', model.get_parameter(w[:-len('weight')] + 'bias'))]]
    layers = [{'class_name': 'Dense', 'config': {'name': w}} for w in linears]
    (directory / 'model.json').write_text(json.dumps({
        'modelTopology': {'model_config': {'config': {'layers': layers}}},
        'weightsManifest': [{'paths': ['shard.bin'], 'weights': weights}]}))
    get_flat().numpy().astype('<f4').tofile(directory / 'shard.bin')
    print(f'exported {len(weights)} weights to {directory}')


def request(url, body=None):
    with urllib.request.urlopen(urllib.request.Request(url, body), timeout=60) as response:
        return response.status, response.headers, response.read()


def wait_for(url, ready, poll_s=1.0):
    """GETs url (JSON) until ready(answer); S0 or S1 may still be starting. Twice party.TIMEOUT_S covers a round that
    fails and the next one, so running out means a server is stuck."""
    deadline, error = time.monotonic() + 2 * party.TIMEOUT_S, None
    while time.monotonic() < deadline:
        try:
            answer = json.loads(request(url)[2])
            if ready(answer):
                return answer
        except OSError as e:
            error = e
        time.sleep(poll_s)
    raise TimeoutError(f'{url} still not ready, last error: {error}')


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f'n{CLIENTS_PER_ROUND}-a{ATTACKERS}.jsonl'
    out.unlink(missing_ok=True)
    weights = get_flat().numpy()
    indices = party.indices(len(weights))
    oracle = EPPRFL(None, filter_lambda=0.005)
    oracle.indices = indices
    print(f'round 0: {fmt(evaluate(torch.from_numpy(weights)))}', flush=True)
    timestamp, mismatches = None, 0

    for round_number in range(1, N_ROUNDS + 1):
        timestamp = wait_for(f'{S0_URL}/round', lambda r: r['open'] and r['timestamp'] != timestamp)['timestamp']
        # all clients: in order, so the attackers are clients 0..ATTACKERS-1 every round, as in robustness-evaluation
        clients = (np.arange(N_CLIENTS) if CLIENTS_PER_ROUND == N_CLIENTS
                   else rng.choice(N_CLIENTS, CLIENTS_PER_ROUND, replace=False))
        # S0 keys the seed by the uuid's tag, so the upload has to use the same uuid's
        ids, seeds = [uuid.uuid4().hex for _ in clients], []
        for id_ in ids:
            status, headers, body = request(f'{S0_URL}/model?uuid={id_}&timestamp=0')
            if status != 200 or int(headers['X-Timestamp']) != timestamp:
                raise RuntimeError(f'round {timestamp} closed while fetching the model')
            served = np.frombuffer(body, '<f4')
            if not np.array_equal(served, weights):
                raise RuntimeError('S0 serves another model than expected; stale state? docker compose ... down -v')
            seeds.append(bytes.fromhex(headers['X-Seed']))

        global_weights = torch.from_numpy(weights.copy())
        # every client trains: the attackers flip the mean of their own honest updates
        updates = np.zeros((len(clients), len(weights)), np.float32)
        for row, client in enumerate(clients):
            updates[row] = local_train(global_weights, partitions[client]).numpy()
        updates, malicious = SignFlippingAttack(ATTACKERS).perform_attack(updates, None)
        updates = updates.astype(np.float32)
        bodies = [shares.client_upload(seed, update) for seed, update in zip(seeds, updates)]

        # a window with no upload is reopened by S1 under the same timestamp, so wait for one with time left
        window = wait_for(f'{S1_URL}/round', lambda w: w and w['timestamp'] == timestamp
                          and w['closes'] - time.time() > UPLOAD_MARGIN_S)
        for id_, body in zip(ids, bodies):
            request(f'{S1_URL}/upload?client={shares.client_tag(id_)}&timestamp={timestamp}', body)
        if json.loads(request(f'{S1_URL}/round')[2]) != window:
            raise RuntimeError("uploads straddled S1's window; raise ROUND_SECONDS")

        start = time.perf_counter()
        plain, plain_benign = oracle.aggregate(updates, None)
        plain_s = time.perf_counter() - start
        plain_clipped = oracle.last_info.get('epprfl_n_clipped', 0)
        wait_for(f'{S0_URL}/round', lambda r: r['open'] and r['timestamp'] != timestamp, poll_s=0.2)
        server = json.loads(request(f'{S0_URL}/stats')[2])
        if server.get('timestamp') != timestamp or server.get('failed'):
            raise RuntimeError(f'S0 did not aggregate round {timestamp}: {server}')
        new = np.frombuffer(request(f'{S0_URL}/model?uuid=compare&timestamp=0')[2], '<f4')
        expected = (weights + plain).astype(np.float32)
        difference = float(np.abs(new - expected).max())
        mismatch = difference > TOLERANCE or (server['benign'], server['clipped']) != (len(plain_benign), plain_clipped)
        mismatches += mismatch

        p = len(weights)
        row = {'round': round_number, 'n': CLIENTS_PER_ROUND, 'attackers': ATTACKERS, 'p': p,
               'benign_plain': len(plain_benign), 'benign_secure': server['benign'],
               'clipped_plain': plain_clipped, 'clipped_secure': server['clipped'],
               'attackers_accepted': int(np.isin(malicious, plain_benign).sum()),
               'max_abs_diff': difference, 'mismatch': bool(mismatch),
               'plain_aggregate_s': plain_s, **{k: server[k] for k in [
                   'server_round_s', 'compile_s', 'party_wall_s', 'mpc_s', 'glue_s',
                   'sent_mb', 'global_sent_mb', 'mpc_rounds']},
               # per client: float32 model down; uint64 shares up, against float32 updates in plaintext FL
               'client_down_mb': 4 * p / 1e6, 'client_up_mb': 8 * p / 1e6, 'plain_client_up_mb': 4 * p / 1e6,
               'plain': evaluate(torch.from_numpy(expected)), 'secure': evaluate(torch.from_numpy(new.copy()))}
        with out.open('a') as f:
            f.write(json.dumps(row) + '\n')

        print(f'round {round_number}: benign plain/secure {row["benign_plain"]}/{row["benign_secure"]}, '
              f'clipped {plain_clipped}/{server["clipped"]}, attackers accepted {row["attackers_accepted"]}, '
              f'max |secure - plain| {difference:.2e}{"  MISMATCH" if mismatch else ""}', flush=True)
        print(f'  time: plain {plain_s:.2f}s, secure {server["server_round_s"]:.1f}s (compile {server["compile_s"]:.1f}s, '
              f'party 0 {server["party_wall_s"]:.1f}s), MPC sent {server["sent_mb"]} MB/party '
              f'in {server["mpc_rounds"]} rounds, global {server["global_sent_mb"]} MB', flush=True)
        print(f'  plain:  {fmt(row["plain"])}')
        print(f'  secure: {fmt(row["secure"])}', flush=True)
        weights = new.copy()
    return mismatches


if __name__ == '__main__':
    if sys.argv[1:2] == ['export']:
        export(Path(sys.argv[2]))
    else:
        mismatches = run()
        print(f'{mismatches} of {N_ROUNDS} rounds differ from the oracle')
        sys.exit(1 if mismatches else 0)
