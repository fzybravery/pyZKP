from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from crypto.ecc.bn254 import (
    G1,
    G1_GENERATOR,
    G1_ZERO,
    G2,
    G2_GENERATOR,
    g1_mul,
    g1_sub,
    g2_add,
    g2_mul,
)
from crypto.field.fr import FR_MODULUS, fr_rand
from crypto.msm import msm_naive_g1
from crypto.poly import poly_eval


@dataclass(frozen=True)
class SRS:
    g1_powers: Tuple[G1, ...]
    g2_powers: Tuple[G2, ...]

# 生成 srs
def setup_srs(max_degree: int) -> SRS:
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    tau = fr_rand(nonzero=True)
    g1: List[G1] = []
    g2: List[G2] = []
    cur = 1
    for _ in range(max_degree + 1):
        g1.append(g1_mul(G1_GENERATOR, cur))
        g2.append(g2_mul(G2_GENERATOR, cur))
        cur = (cur * tau) % FR_MODULUS
    return SRS(g1_powers=tuple(g1), g2_powers=tuple(g2))

# 生成 kzg 承诺
def commit(srs: SRS, coeffs: Sequence[int]) -> G1:
    if len(coeffs) == 0:
        return G1_ZERO
    if len(coeffs) > len(srs.g1_powers):
        raise ValueError("SRS too small for polynomial degree")
    scalars = [int(c) % FR_MODULUS for c in coeffs]
    return msm_naive_g1(srs.g1_powers[: len(scalars)], scalars)

# 在域上计算多项式除法，返回商多项式系数列表
def _synthetic_division(coeffs: Sequence[int], z: int) -> List[int]:
    if len(coeffs) <= 1:
        return []
    z = int(z) % FR_MODULUS
    out = [0] * (len(coeffs) - 1)
    acc = coeffs[-1] % FR_MODULUS
    out[-1] = acc
    for i in range(len(coeffs) - 2, 0, -1):
        acc = (coeffs[i] + acc * z) % FR_MODULUS
        out[i - 1] = acc
    return out

# 打开证明
def open_proof(srs: SRS, coeffs: Sequence[int], z: int) -> Tuple[int, G1]:
    zz = int(z) % FR_MODULUS
    y = poly_eval(coeffs, zz)
    f0 = list(coeffs)
    f0[0] = (f0[0] - y) % FR_MODULUS
    q = _synthetic_division(f0, zz)
    pi = commit(srs, q)
    return y, pi

# 验证证明
def verify_proof(srs: SRS, commitment: G1, z: int, y: int, proof: G1) -> bool:
    from py_ecc import optimized_bn128 as b

    zz = int(z) % FR_MODULUS
    yy = int(y) % FR_MODULUS
    g2_tau = srs.g2_powers[1]
    g2_0 = srs.g2_powers[0]
    q = g2_add(g2_tau, g2_mul(g2_0, (-zz) % FR_MODULUS))  # (tau - z)G2
    left = b.pairing(q, proof)
    right = b.pairing(g2_0, g1_sub(commitment, g1_mul(G1_GENERATOR, yy)))
    return left == right
