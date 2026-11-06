"""Compiling and running epprfl.mpc in an MP-SPDZ directory, for S0, S1 and the tests."""
import hashlib
import logging
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from shares import encode

MPSPDZ = Path(os.environ.get('MPSPDZ', '/usr/src/MP-SPDZ'))
SOURCE = Path(__file__).with_name('epprfl.mpc')
PORT = int(os.environ.get('MPSPDZ_PORT', 5000))
# where party 0 listens, for both parties: MP-SPDZ hands out the address party 0 itself connected from
S0_HOST = os.environ.get('S0_HOST', 'localhost')
# -F: preprocessing from files, for the online-only timings of an INSECURE build (Scripts/setup-online.sh first)
EXTRA = os.environ.get('MPSPDZ_FLAGS', '').split()
# shortcut: fixed at 1 h (the 200-client rounds take minutes), raise it if a slower machine gets near it
TIMEOUT_S = 3600

log = logging.getLogger('party')


def indices(p: int) -> np.ndarray:
    """R_idx: eps * p distinct coordinates, the same on both servers from RIDX_SEED; eps = 0.06 as in the paper."""
    rng = np.random.default_rng(int(os.environ.get('RIDX_SEED', 0)))
    return np.sort(rng.choice(p, int(0.06 * p), replace=False))


def compile_program(n: int, p: int, k: int) -> str:
    """Compiles once per (source, N, P, K). Deterministic, so both servers get the same bytecode."""
    # the source hash in the name: an edited epprfl.mpc never runs stale bytecode, and the cache keeps the two local
    # servers from rewriting the bytecode every round while the other party loads it
    program = 'epprfl_' + hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:8]
    name = f'{program}-{n}-{p}-{k}'
    if not (MPSPDZ / 'Programs' / 'Schedules' / f'{name}.sch').exists():
        shutil.copy(SOURCE, MPSPDZ / 'Programs' / 'Source' / f'{program}.mpc')
        start = time.perf_counter()
        subprocess.run(['./compile.py', '-M', '-R', '64', program, str(n), str(p), str(k)], cwd=MPSPDZ, check=True,
                       stdout=subprocess.DEVNULL)
        log.info('compiled %s in %.1fs', name, time.perf_counter() - start)
    return name


def init_state(player: int):
    """Before the first round: MP-SPDZ's Z2^64 header and shares of zero for the 4 state values of epprfl.mpc."""
    path = MPSPDZ / 'Persistence' / f'Transactions-P{player}.data'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(struct.pack('<Q', 9) + b'Z2^64' + struct.pack('<i', 64) + bytes(4 * 8))


def write_input(player: int, shares: list, indices: np.ndarray, history: np.ndarray | None = None):
    """The player's Input-Binary file in epprfl.mpc's order; history is D_h on R_idx, from S0 only."""
    chunks = [share[indices] for share in shares] + ([encode(history)] if player == 0 else []) + shares
    with open(MPSPDZ / 'Player-Data' / f'Input-Binary-P{player}-0', 'wb') as f:
        for chunk in chunks:
            f.write(np.ascontiguousarray(chunk, '<u8').tobytes())


def run(player: int, name: str) -> dict:
    """Runs the party to completion, or raises after TIMEOUT_S if the other party never comes; the stats are MP-SPDZ's
    own lines (time, data sent, rounds)."""
    start = time.perf_counter()
    result = subprocess.run(['./semi2k-party.x', '-p', str(player), '-N', '2', '-h', S0_HOST, '-pn', str(PORT),
                             *EXTRA, name], cwd=MPSPDZ, capture_output=True, text=True, timeout=TIMEOUT_S)
    stats = {'wall_s': time.perf_counter() - start,
             'mpspdz': [line for line in result.stderr.splitlines()
                        if re.match(r'(Time|Data sent|Global data sent|Spent)', line)]}
    if result.returncode != 0:
        raise RuntimeError(f'party {player} failed:\n{result.stdout}\n{result.stderr}')
    log.info('party %d: %s', player, stats)
    return stats


def read_output(n: int, p: int):
    """P0's output: benign bits, then (None if none is benign) clipped bits and the clipped benign sum at 2^40."""
    out = np.fromfile(MPSPDZ / 'Player-Data' / 'Binary-Output-P0-0', '<i8')
    benign = out[:n].astype(bool)
    if not benign.any():
        return benign, None, None
    return benign, out[n:2 * n].astype(bool), out[2 * n:2 * n + p]
