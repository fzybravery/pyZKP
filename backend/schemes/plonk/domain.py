from __future__ import annotations

import math
import secrets
from typing import List, Tuple

from common.crypto.field.fr import FR_MODULUS, fr_inv


ROOT_2_28 = 19103219067921713944291392827692070036145651957329286315305642004821462161904
MAX_ORDER_LOG = 28

# 获取大于等于 n 的最小 2 的幂次
def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

# 计算大小为 n 的域的生成元（n次本原单位根 omega）。
def omega_for_domain(n: int) -> int:
    nn = next_power_of_two(n)
    logn = int(math.log2(nn))
    if 2**logn != nn:
        raise ValueError("n must be power of two")
    if logn > MAX_ORDER_LOG:
        raise ValueError("domain too large for BN254 root of unity")
    expo = 1 << (MAX_ORDER_LOG - logn)
    return pow(ROOT_2_28, expo, FR_MODULUS)

#  从 1 开始，每次乘以生成元 omega，依次生成 [1, w, w^2, ..., w^(n-1)]
def roots_of_unity(n: int, omega: int) -> Tuple[int, ...]:
    roots: List[int] = []
    cur = 1
    for _ in range(n):
        roots.append(cur)
        cur = (cur * omega) % FR_MODULUS
    return tuple(roots)

# 为 PLONK 的置换论证（Permutation Argument）生成陪集偏移因子 k1 和 k2
def find_coset_factors(n: int) -> Tuple[int, int, int]:
    omega = omega_for_domain(n)
    def in_subgroup(x: int) -> bool:
        return pow(x % FR_MODULUS, n, FR_MODULUS) == 1

    while True:
        k1 = secrets.randbelow(FR_MODULUS - 1) + 1
        k2 = secrets.randbelow(FR_MODULUS - 1) + 1
        if k1 == k2 or k1 == 1 or k2 == 1:
            continue
        if in_subgroup(k1) or in_subgroup(k2):
            continue
        if in_subgroup((k1 * fr_inv(k2)) % FR_MODULUS):
            continue
        return omega, k1, k2

# 生成一个随机的陪集偏移因子 (Coset Shift) g。
def coset_shift(n: int) -> int:
    while True:
        g = secrets.randbelow(FR_MODULUS - 1) + 1
        if pow(g, n, FR_MODULUS) != 1:
            return g
