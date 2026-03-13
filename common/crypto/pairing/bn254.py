from __future__ import annotations

from typing import Any, Iterable, Tuple

from py_ecc import optimized_bn128 as b

from pyZKP.common.crypto.ecc.bn254 import G1, G2


def pairing_g1_g2(p: G1, q: G2) -> Any:
    return b.pairing(q, p)


def pairing_prod(pairs: Iterable[Tuple[G1, G2]]) -> Any:
    acc = b.FQ12.one()
    for p, q in pairs:
        acc *= b.pairing(q, p)
    return acc
