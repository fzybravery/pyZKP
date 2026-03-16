from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.ecc.bn254 import G1
from py_ecc import optimized_bn128 as b


def _i2b(x: int) -> bytes:
    return int(x).to_bytes(32, "big", signed=False)


def _g1_to_bytes(p: G1) -> bytes:
    x, y = b.normalize(p)[:2]
    return _i2b(int(x.n)) + _i2b(int(y.n))


@dataclass
class Transcript:
    state: "hashlib._Hash"

    def __init__(self, label: bytes = b"pyZKP-plonk") -> None:
        self.state = hashlib.sha256()
        self.state.update(label)

    def absorb_bytes(self, data: bytes) -> None:
        self.state.update(data)

    def absorb_int(self, x: int) -> None:
        self.state.update(_i2b(x % FR_MODULUS))

    def absorb_g1(self, p: G1) -> None:
        self.state.update(_g1_to_bytes(p))

    def challenge(self, label: bytes) -> int:
        h = self.state.copy()
        h.update(label)
        d = h.digest()
        x = int.from_bytes(d, "big") % FR_MODULUS
        self.state.update(d)
        return x
