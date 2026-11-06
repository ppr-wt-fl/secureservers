"""S0 and S1 on this machine with a small model: clients poll, upload, and the next model is the oracle's."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
import pytest

import party
import shares
from test_epprfl import EPPRFL

N = 6


def get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, dict(response.headers), response.read()


def test_round(tmp_path):
    kernel, bias = (50, 20), (20,)
    model = np.random.default_rng(1).normal(size=50 * 20 + 20).astype(np.float32)
    (tmp_path / 'model.json').write_text(json.dumps({
        'modelTopology': {'model_config': {'config': {'layers': [
            {'class_name': 'InputLayer', 'config': {'name': 'input'}},
            {'class_name': 'Dense', 'config': {'name': 'dense'}}]}}},
        'weightsManifest': [{'paths': ['shard.bin'], 'weights': [
            {'name': 'dense/kernel', 'shape': kernel, 'dtype': 'float32'},
            {'name': 'dense/bias', 'shape': bias, 'dtype': 'float32'}]}]}))
    model.tofile(tmp_path / 'shard.bin')
    for name in ['Transactions-P0.data', 'Transactions-P1.data', 's0.npz']:
        (party.MPSPDZ / 'Persistence' / name).unlink(missing_ok=True)

    env = {**os.environ, 'MODEL_DIR': str(tmp_path), 'ROUND_SECONDS': '3', 'S0_URL': 'http://localhost:8100',
           'S1_TOKEN': 'test-token'}
    servers = [subprocess.Popen([sys.executable, '-m', 'uvicorn', f's{i}:app', '--port', str(8100 + i)], env=env)
               for i in [0, 1]]
    try:
        for _ in range(50):
            try:
                first = json.loads(get('http://localhost:8100/round')[2])
                break
            except OSError:
                time.sleep(0.2)
        arrived = json.dumps({'timestamp': first['timestamp'], 'clients': []}).encode()
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(urllib.request.Request('http://localhost:8100/arrived', arrived,
                                                          {'Content-Type': 'application/json'}))
        assert refused.value.code == 403
        rng = np.random.default_rng(2)
        updates = rng.normal(size=(N, len(model)))
        updates = (updates / np.linalg.norm(updates, axis=1, keepdims=True) * 1.58).astype(np.float32)
        for update in updates:
            client = uuid.uuid4().hex
            status, headers, body = get(f'http://localhost:8100/model?uuid={client}&timestamp=0')
            assert np.array_equal(np.frombuffer(body, '<f4'), model)
            # up to date: nothing to fetch
            assert get(f'http://localhost:8100/model?uuid={client}&timestamp={headers["x-timestamp"]}')[0] == 204
            # what S1 sees of the client does not fetch the client's seed
            tag = shares.client_tag(client)
            assert get(f'http://localhost:8100/model?uuid={tag}&timestamp=0')[1]['x-seed'] != headers['x-seed']
            request = urllib.request.Request(
                f'http://localhost:8101/upload?client={tag}&timestamp={headers["x-timestamp"]}',
                shares.client_upload(bytes.fromhex(headers['x-seed']), update))
            urllib.request.urlopen(request)

        for _ in range(100):
            info = json.loads(get('http://localhost:8100/round')[2])
            if info['open'] and info['timestamp'] != first['timestamp']:
                break
            time.sleep(0.5)
        new = np.frombuffer(get(f'http://localhost:8100/model?uuid=x&timestamp=0')[2], '<f4')
        oracle = EPPRFL(None, filter_lambda=0.005)
        oracle.indices = party.indices(len(model))
        expected, benign = oracle.aggregate(updates, None)
        assert len(benign) > 0
        assert np.abs(new - (model + expected)).max() < 1e-3
    finally:
        for server in servers:
            server.terminate()
