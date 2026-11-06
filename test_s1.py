"""Which uploads S1 keeps. Without MP-SPDZ: the rounds thread only starts with the app's lifespan, which stays off here."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault('S1_TOKEN', 'test-token')
import s1  # noqa: E402

P = 4
GOOD = bytes(8 * P)
client = TestClient(s1.app)


@pytest.fixture(autouse=True)
def window(monkeypatch):
    monkeypatch.setattr(s1, 'current', {'timestamp': 7, 'p': P, 'bodies': {}, 'closes': 0})


def upload(name, body=GOOD, timestamp=7):
    assert client.post('/upload', params={'client': name, 'timestamp': timestamp}, content=body).status_code == 204


def test_keeps_the_right_upload_once():
    upload('a')
    upload('a', b'\1' * (8 * P))
    assert s1.current['bodies'] == {'a': GOOD}


@pytest.mark.parametrize('body, timestamp', [(GOOD[:-1], 7), (GOOD + b'\0', 7), (GOOD * 1000, 7), (b'', 7), (GOOD, 6)])
def test_drops_wrong_length_and_round(body, timestamp):
    upload('a', body, timestamp)
    assert s1.current['bodies'] == {}


def test_drops_when_closed(monkeypatch):
    monkeypatch.setattr(s1, 'current', None)
    upload('a')
    assert s1.current is None


def test_drops_over_the_client_limit(monkeypatch):
    monkeypatch.setattr(s1, 'MAX_CLIENTS', 2)
    for name in 'abc':
        upload(name)
    assert list(s1.current['bodies']) == ['a', 'b']
