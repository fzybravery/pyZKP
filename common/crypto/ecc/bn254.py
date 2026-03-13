from __future__ import annotations

from typing import Any, Tuple

from py_ecc import optimized_bn128 as b

G1 = Tuple[Any, Any, Any]
G2 = Tuple[Any, Any, Any]


def g1_add(p: G1, q: G1) -> G1:
    return b.add(p, q)


def g1_neg(p: G1) -> G1:
    return b.neg(p)


def g1_sub(p: G1, q: G1) -> G1:
    return b.add(p, b.neg(q))


def g1_mul(p: G1, s: int) -> G1:
    return b.multiply(p, int(s))


def g1_eq(p: G1, q: G1) -> bool:
    return b.normalize(p) == b.normalize(q)


def g2_add(p: G2, q: G2) -> G2:
    return b.add(p, q)


def g2_neg(p: G2) -> G2:
    return b.neg(p)


def g2_mul(p: G2, s: int) -> G2:
    return b.multiply(p, int(s))


def g2_eq(p: G2, q: G2) -> bool:
    return b.normalize(p) == b.normalize(q)


G1_GENERATOR: G1 = b.G1
G2_GENERATOR: G2 = b.G2
G1_ZERO: G1 = b.Z1
G2_ZERO: G2 = b.Z2
