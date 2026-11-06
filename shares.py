"""Client-side additive sharing over Z_2^64, as the extension does it (src/lib/fl/shares.ts there).

The client gets a 16-byte seed r_i from S0, expands it to its share L_i and uploads (encode(update) - L_i) mod 2^64
to S1, so S0 and S1 hold additive shares of the encoded update.
"""
import hashlib

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

SCALE = 2 ** 20


def client_tag(uuid: str) -> str:
    """What S1 and the arrived list know a client by. The uuid itself fetches the client's seed from S0, so S1 must not
    see it: with the seed and the upload it could open the update."""
    return hashlib.sha256(uuid.encode()).hexdigest()


def expand(seed: bytes, n: int) -> np.ndarray:
    """L_i: AES-128-CTR with the seed as key and a zero counter block, over 8n zero bytes, as n little-endian uint64."""
    encryptor = Cipher(algorithms.AES(seed), modes.CTR(bytes(16))).encryptor()
    return np.frombuffer(encryptor.update(bytes(8 * n)) + encryptor.finalize(), '<u8')


def encode(values) -> np.ndarray:
    """round-half-up(v * 2^20) mod 2^64. Not np.round: banker's rounding differs from Math.round on exact ties."""
    return np.floor(np.asarray(values, np.float64) * SCALE + 0.5).astype(np.int64).view(np.uint64)


def decode(values) -> np.ndarray:
    return np.asarray(values, np.uint64).view(np.int64) / SCALE


def client_upload(seed: bytes, update) -> bytes:
    """The body a client sends to S1 (wraps mod 2^64, as uint64 arithmetic does)."""
    return (encode(update) - expand(seed, len(update))).astype('<u8').tobytes()
