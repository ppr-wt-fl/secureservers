"""S0: hands out the global model and the clients' seeds r_i, runs MP-SPDZ party 0 and publishes the next model.

Run: uvicorn s0:app --port 8000. MODEL_DIR holds the extension's model.json and weight shard (the first model).
"""
import hmac
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

import party
import shares

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('s0')

MODEL_DIR = Path(os.environ.get('MODEL_DIR', '/model'))
# next to MP-SPDZ's own state (Persistence/Transactions-P0.data), both have to stay in step
STATE = party.MPSPDZ / 'Persistence' / 's0.npz'
# proves a call to /arrived comes from S1; the rest of the port is public, for the clients
S1_TOKEN = os.environ['S1_TOKEN'].encode()


def initial_model() -> np.ndarray:
    """The weight shard, flat in manifest order, which has to be the order of model.trainableWeights the client
    sends its update in: [kernel, bias] per Dense layer in topology order, no other layer with weights."""
    spec = json.loads((MODEL_DIR / 'model.json').read_text())
    layers = spec['modelTopology']['model_config']['config']['layers']
    trainable = [f'{layer["config"]["name"]}/{w}' for layer in layers if layer['class_name'] == 'Dense'
                 for w in ('kernel', 'bias')]
    weights = [w for group in spec['weightsManifest'] for w in group['weights']]
    if [w['name'] for w in weights] != trainable or any(w['dtype'] != 'float32' for w in weights):
        raise ValueError(f'weight manifest {[w["name"] for w in weights]} is not the trainable order {trainable}')
    paths = [path for group in spec['weightsManifest'] for path in group['paths']]
    flat = np.concatenate([np.fromfile(MODEL_DIR / path, '<f4') for path in paths])
    if len(flat) != sum(int(np.prod(w['shape'])) for w in weights):
        raise ValueError('weight shard size does not match the manifest')
    return flat


if STATE.exists():
    with np.load(STATE) as saved:
        model, history_sum, count = saved['model'], saved['history_sum'], int(saved['count'])
else:
    model = initial_model()
    history_sum, count = np.zeros(len(party.indices(len(model)))), 0
    party.init_state(0)
P = len(model)
INDICES = party.indices(P)
# the open round: clients get the seed with the model, S1 closes the round by posting who uploaded
current = {'timestamp': time.time_ns() // 10 ** 6, 'seeds': {}, 'open': True}
lock = threading.Lock()
# the last aggregated round's counts and costs, for the evaluation (mltest/compare.py)
last_round = {}

app = FastAPI()


def publish(new_model: np.ndarray):
    """Starts the next round, with a new timestamp so every client fetches the model and a fresh seed."""
    global model
    model = new_model
    np.savez(STATE.with_suffix('.tmp.npz'), model=model, history_sum=history_sum, count=count)
    os.replace(STATE.with_suffix('.tmp.npz'), STATE)
    with lock:
        current.update(timestamp=time.time_ns() // 10 ** 6, seeds={}, open=True)
    log.info('round %d open, %d rounds aggregated', current['timestamp'], count)


@app.get('/model')
def poll(uuid: str, timestamp: int = 0):
    with lock:
        if timestamp == current['timestamp'] or not current['open']:
            return Response(status_code=204)
        # the same seed again if this client asks twice in a round, e.g. after a restart
        seed = current['seeds'].setdefault(shares.client_tag(uuid), os.urandom(16))
        ts = current['timestamp']
    return Response(model.astype('<f4').tobytes(), media_type='application/octet-stream',
                    headers={'X-Timestamp': str(ts), 'X-Seed': seed.hex()})


@app.get('/round')
def round_info():
    """For S1: the round it collects uploads for, and the update length."""
    return {'timestamp': current['timestamp'], 'open': current['open'], 'p': P}


@app.get('/stats')
def stats():
    """The last aggregated round: benign and clipped counts, compile, MPC and glue time, MP-SPDZ's traffic."""
    return last_round


def mpspdz_numbers(lines: list[str]) -> dict:
    """Party 0's MP-SPDZ summary lines as numbers (None where a line is missing)."""
    text = '\n'.join(lines)
    patterns = {'mpc_s': r'^Time = ([\d.e+-]+)', 'sent_mb': r'^Data sent = ([\d.e+-]+) MB',
                'mpc_rounds': r'^Data sent = .* in ~(\d+) rounds', 'global_sent_mb': r'^Global data sent = ([\d.e+-]+)'}
    return {key: float(m.group(1)) if (m := re.search(pattern, text, re.M)) else None
            for key, pattern in patterns.items()}


class Arrived(BaseModel):
    timestamp: int
    clients: list[str]


@app.post('/arrived')
def arrived(body: Arrived, tasks: BackgroundTasks, x_s1_token: str = Header('')):
    """S1 closes the round. Returns the uploads S0 has a seed for, in the order both parties input them."""
    if not hmac.compare_digest(x_s1_token.encode(), S1_TOKEN):
        raise HTTPException(403, 'not S1')
    with lock:
        if body.timestamp != current['timestamp'] or not current['open']:
            raise HTTPException(409, 'not the open round')

        # keep S1's order, since both parties input the clients in it; drop clients without a seed and repeats
        clients = []
        for client in body.clients:
            if client in current['seeds'] and client not in clients:
                clients.append(client)
        seeds = [current['seeds'][client] for client in clients]
        # no more seeds are handed out for this round
        current['open'] = False
    if clients:
        tasks.add_task(aggregate, seeds)
    else:
        publish(model)
    return {'clients': clients}


def aggregate(seeds: list[bytes]):
    global history_sum, count, last_round
    begin = start = time.perf_counter()
    n = len(seeds)
    name = party.compile_program(n, P, len(INDICES))
    compile_s = time.perf_counter() - start
    party.write_input(0, [shares.expand(seed, P) for seed in seeds], INDICES,
                      history_sum / count if count else np.zeros(len(INDICES)))
    glue = time.perf_counter() - start
    try:
        stats = party.run(0, name)
    except Exception:
        log.exception('party 0 failed, the round is reopened with the same model')
        last_round = {'timestamp': current['timestamp'], 'n': n, 'failed': True}
        publish(model)
        return
    start = time.perf_counter()
    benign, clipped, total = party.read_output(n, P)
    timing = {'n': n, **stats, 'glue_s': glue + time.perf_counter() - start}
    last_round = {'timestamp': current['timestamp'], 'n': n, 'benign': int(benign.sum()),
                  'clipped': 0 if clipped is None else int(clipped.sum()), 'compile_s': compile_s,
                  'party_wall_s': stats['wall_s'], 'glue_s': timing['glue_s'] - compile_s,
                  **mpspdz_numbers(stats['mpspdz']),
                  # from S1's /arrived to the next round opening, which happens right after this
                  'server_round_s': time.perf_counter() - begin}
    if total is None:
        log.info('no benign client, model and history unchanged')
        publish(model)
    else:
        update = total / 2 ** 40 / benign.sum()
        history_sum = history_sum + update[INDICES]
        count += 1
        log.info('%d of %d benign, %d clipped', benign.sum(), n, clipped.sum())
        publish((model + update).astype(np.float32))
    log.info('timing: %s', timing)
