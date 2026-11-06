"""S1: collects the clients' shares for S0's open round, closes it after ROUND_SECONDS and runs MP-SPDZ party 1.

Run: uvicorn s1:app --port 8001. S0_URL is S0's HTTP address, S0_HOST where party 0 listens.
"""
import json
import logging
import os
import threading
import time
import urllib.request
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request, Response

import party

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('s1')

S0_URL = os.environ.get('S0_URL', 'http://localhost:8000')
ROUND_SECONDS = float(os.environ.get('ROUND_SECONDS', 600))
S1_TOKEN = os.environ['S1_TOKEN']
# shortcut: first come, first served, so fake clients can still fill it; stopping them needs client logins
MAX_CLIENTS = 200

# the round uploads are collected for; None while closed
current = None
lock = threading.Lock()


def s0(path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(S0_URL + path, data, {'Content-Type': 'application/json', 'X-S1-Token': S1_TOKEN})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def rounds():
    """Deadline loop: wait for S0's next open round, collect, close, hand S0 the arrived list, run party 1.
    A round with no upload is collected again under the same timestamp; S0 is not told and the round stays open."""
    global current
    party.init_state(1)
    done = None
    while True:
        try:
            info = s0('/round')
        except OSError as error:
            log.warning('S0 unreachable: %s', error)
            time.sleep(5)
            continue
        if not info['open'] or info['timestamp'] == done:
            time.sleep(1)
            continue
        with lock:
            current = {'timestamp': info['timestamp'], 'p': info['p'], 'bodies': {},
                       'closes': time.time() + ROUND_SECONDS}
        time.sleep(ROUND_SECONDS)
        with lock:
            closed, current = current, None
        if not closed['bodies']:
            continue
        try:
            clients = s0('/arrived', {'timestamp': closed['timestamp'], 'clients': list(closed['bodies'])})['clients']
            if clients:
                p = closed['p']
                indices = party.indices(p)
                name = party.compile_program(len(clients), p, len(indices))
                party.write_input(1, [np.frombuffer(closed['bodies'][u], '<u8') for u in clients], indices)
                party.run(1, name)
        except Exception:
            # S0 reopens the round on its side, at the latest when its party times out waiting for this one
            log.exception('round %d failed', closed['timestamp'])
        done = closed['timestamp']


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=rounds, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get('/round')
def round_info():
    """The window uploads are collected in: S0's timestamp and when it closes (Unix seconds); null while closed."""
    with lock:
        return current and {'timestamp': current['timestamp'], 'closes': current['closes']}


def accepts(client: str, timestamp: int) -> bool:
    return bool(current) and timestamp == current['timestamp'] and client not in current['bodies'] \
        and len(current['bodies']) < MAX_CLIENTS


@app.post('/upload')
async def upload(client: str, timestamp: int, request: Request):
    """Silently drops wrong-round, duplicate, wrong-length and over-limit uploads: the client learns nothing from the
    reply. client is shares.client_tag of the uuid, never the uuid. Stops reading at the expected length, so a body
    of any size costs at most that much memory."""
    with lock:
        limit = 8 * current['p'] if accepts(client, timestamp) else 0
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return Response(status_code=204)
    with lock:
        if accepts(client, timestamp) and len(body) == 8 * current['p']:
            current['bodies'][client] = bytes(body)
    return Response(status_code=204)
