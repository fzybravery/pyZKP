from __future__ import annotations

from typing import Sequence

from common.crypto.ecc.bn254 import G1, G1_ZERO, G2, G2_ZERO, g1_add, g1_mul, g2_add, g2_mul

# g1 群朴素多标量乘法
def msm_naive_g1(points: Sequence[G1], scalars: Sequence[int]) -> G1:
    acc: G1 = G1_ZERO
    for p, s in zip(points, scalars):
        acc = g1_add(acc, g1_mul(p, int(s)))
    return acc

# g2 群朴素多标量乘法
def msm_naive_g2(points: Sequence[G2], scalars: Sequence[int]) -> G2:
    acc: G2 = G2_ZERO
    for p, s in zip(points, scalars):
        acc = g2_add(acc, g2_mul(p, int(s)))
    return acc


def msm_naive(points, scalars):
    if len(points) == 0:
        raise ValueError("empty msm")
    p0 = points[0]
    if isinstance(p0[0], tuple):
        return msm_naive_g2(points, scalars)
    return msm_naive_g1(points, scalars)
