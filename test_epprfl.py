"""epprfl.mpc against the plaintext oracle, both parties on this machine. Needs MPSPDZ set to an MP-SPDZ directory
with semi2k-party.x (semi2k needs no SSL keys)."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import party
import shares

sys.path.insert(0, os.environ.get('ORACLE', str(Path(__file__).parents[2] / 'robustness-evaluation/poisoning-eval')))
from epprfl import EPPRFL  # noqa: E402

N, P = 8, 5000
K = int(0.06 * P)


def secure_round(updates, indices, history):
    """What S0 and S1 do with a round's uploads; returns the oracle's (aggregate, benign) plus the clipped set."""
    seeds = [os.urandom(16) for _ in updates]
    l = [shares.expand(seed, P) for seed in seeds]
    bodies = [np.frombuffer(shares.client_upload(seed, update), '<u8') for seed, update in zip(seeds, updates)]
    name = party.compile_program(N, P, K)
    party.write_input(0, l, indices, history)
    party.write_input(1, bodies, indices)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda player: party.run(player, name), [0, 1]))
    benign, clipped, total = party.read_output(N, P)
    if total is None:
        return np.zeros(P), np.flatnonzero(benign), clipped
    return total / 2 ** 40 / benign.sum(), np.flatnonzero(benign), np.flatnonzero(clipped)


def stored_m_h():
    """m_h from the two parties' persistence shares; only the test opens it."""
    paths = [party.MPSPDZ / 'Persistence' / f'Transactions-P{i}.data' for i in [0, 1]]
    total = sum(int(np.frombuffer(path.read_bytes()[-32:], '<u8')[0]) for path in paths) % 2 ** 64
    return (total - 2 ** 64 if total >= 2 ** 63 else total) / 2 ** 20


def test_against_oracle():
    for i in [0, 1]:
        (party.MPSPDZ / 'Persistence' / f'Transactions-P{i}.data').unlink(missing_ok=True)
        party.init_state(i)
    rng = np.random.default_rng(0)
    oracle = EPPRFL(rng, filter_lambda=0.005)
    oracle.indices = np.sort(rng.choice(P, K, replace=False))

    def honest():
        # norm 1.58 noise (5 at lambda 0.05: d scales with lambda): distances to D_h stay near the first round's m_h, so later rounds accept some clients
        updates = rng.normal(size=(N, P))
        return (updates / np.linalg.norm(updates, axis=1, keepdims=True) * 1.58).astype(np.float32)

    sign_flip = honest()
    sign_flip[:2] *= -3
    rounds = {'first': honest(), 'later': honest(), 'skipped': -20 * honest(), 'sign flip': sign_flip}

    history_sum, count = np.zeros(K), 0
    for label, updates in rounds.items():
        m_h_before = oracle.m_h
        expected, expected_benign = oracle.aggregate(updates, None)
        history = history_sum / count if count else np.zeros(K)
        aggregate, benign, clipped = secure_round(updates, oracle.indices, history)

        assert list(benign) == list(expected_benign), label
        if len(benign) == 0:
            assert clipped is None and oracle.m_h == m_h_before
            continue
        distances = np.array(oracle.last_info['epprfl_distances'])
        m_h = m_h_before if m_h_before is not None else np.median(distances)
        assert list(clipped) == [i for i in benign if distances[i] > m_h], label
        assert np.abs(aggregate - expected).max() < 1e-3, label
        assert stored_m_h() == pytest.approx(oracle.m_h, rel=1e-3), label
        history_sum += aggregate[oracle.indices]
        count += 1
    # every branch ran: clipping, a skip, a later round, the attackers filtered out
    assert len(clipped) > 0 and count == 3 and not {0, 1} & set(benign)
