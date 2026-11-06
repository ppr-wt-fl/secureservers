import numpy as np

import shares

# the same vector is in the extension's src/lib/fl/shares.test.ts
SEED = bytes(range(16))
# the last two are exact ties: half-up gives 2 and -1, banker's rounding 2 and -2
UPDATE = np.array([0.5, -1.25, 1.5 / 2 ** 20, -1.5 / 2 ** 20], np.float32)
TAG = 'd5fe82513196ca977d8166092b9a47dc27ce80a93dd16f6680f2111ef703c268'
UPLOAD = '3a5eccc87870a47d91b06a9d5e3727868fb9ec6a6a3f4be1b684421c9a0bd2f5'


def test_client_tag():
    assert shares.client_tag('test-client') == TAG


def test_vector():
    assert list(shares.encode(UPDATE).view(np.int64)) == [524288, -1310720, 2, -1]
    body = shares.client_upload(SEED, UPDATE)
    assert body.hex() == UPLOAD
    # S0's share plus S1's gives the encoded update back
    total = shares.expand(SEED, 4) + np.frombuffer(body, '<u8')
    assert np.array_equal(total, shares.encode(UPDATE))
