from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from pyZKP.common.crypto.field.fr import FR_MODULUS, fr_inv

ROOT_2_28 = 19103219067921713944291392827692070036145651957329286315305642004821462161904
MAX_ORDER_LOG = 28

# 计算 n 次单位根 omega
def omega_for_size(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if n & (n - 1) != 0:
        raise ValueError("n must be a power of two")
    logn = int(math.log2(n))
    if (1 << logn) != n:
        raise ValueError("n must be a power of two")
    if logn > MAX_ORDER_LOG:
        raise ValueError("domain too large for BN254 root of unity")
    expo = 1 << (MAX_ORDER_LOG - logn)
    return pow(ROOT_2_28, expo, FR_MODULUS)


# 计算 n 次单位根 omega 的所有次方
def roots_of_unity(n: int, omega: int) -> Tuple[int, ...]:
    if n <= 0:
        raise ValueError("n must be positive")
    if n & (n - 1) != 0:
        raise ValueError("n must be a power of two")
    roots: List[int] = []
    cur = 1
    for _ in range(n):
        roots.append(cur)
        cur = (cur * (omega % FR_MODULUS)) % FR_MODULUS
    return tuple(roots)


# 原地执行 NTT （系数 -> 值）
def ntt_inplace(a: List[int], omega: int) -> None:
    n = len(a)
    if n == 0:
        return
    if n & (n - 1) != 0:
        raise ValueError("length must be power of two")

    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = pow(int(omega) % FR_MODULUS, n // length, FR_MODULUS)
        for i in range(0, n, length):
            w = 1
            half = length >> 1
            for j in range(half):
                u = a[i + j] % FR_MODULUS
                v = (a[i + j + half] % FR_MODULUS) * w % FR_MODULUS
                a[i + j] = (u + v) % FR_MODULUS
                a[i + j + half] = (u - v) % FR_MODULUS
                w = (w * wlen) % FR_MODULUS
        length <<= 1


# 原地执行 INTT （值 -> 系数）
def intt_inplace(a: List[int], omega: int) -> None:
    n = len(a)
    if n == 0:
        return
    inv_omega = fr_inv(int(omega) % FR_MODULUS)
    ntt_inplace(a, inv_omega)
    inv_n = fr_inv(n)
    for i in range(n):
        a[i] = (a[i] % FR_MODULUS) * inv_n % FR_MODULUS


# NTT 返回新列表
def ntt(coeffs: Sequence[int], omega: int) -> List[int]:
    out = [int(x) % FR_MODULUS for x in coeffs]
    ntt_inplace(out, omega)
    return out

# INTT 返回新列表
def intt(evals: Sequence[int], omega: int) -> List[int]:
    out = [int(x) % FR_MODULUS for x in evals]
    intt_inplace(out, omega)
    return out

# 给定多项式系数，求其在 n 次单位根上的点值
def evals_from_coeffs_on_roots(coeffs: Sequence[int], n: int, omega: int) -> List[int]:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    out = [0] * n
    for i in range(min(n, len(coeffs))):
        out[i] = int(coeffs[i]) % FR_MODULUS
    ntt_inplace(out, omega)
    return out

# 给定多项式值，求其系数
def coeffs_from_evals_on_roots(evals: Sequence[int], omega: int) -> List[int]:
    if len(evals) == 0 or (len(evals) & (len(evals) - 1)) != 0:
        raise ValueError("len(evals) must be a power of two")
    out = [int(x) % FR_MODULUS for x in evals]
    intt_inplace(out, omega)
    return out

# 在陪集上，给定多项式系数，求其在陪集上的点值
def evals_from_coeffs_on_coset(coeffs: Sequence[int], *, n: int, omega: int, shift: int) -> List[int]:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    ss = int(shift) % FR_MODULUS
    out = [0] * n
    pow_s = 1
    for i in range(min(n, len(coeffs))):
        out[i] = (int(coeffs[i]) % FR_MODULUS) * pow_s % FR_MODULUS
        pow_s = (pow_s * ss) % FR_MODULUS
    ntt_inplace(out, omega)
    return out


# 在陪集上，给定多项式值，求其系数
def coeffs_from_evals_on_coset(evals: Sequence[int], *, omega: int, shift: int) -> List[int]:
    if len(evals) == 0 or (len(evals) & (len(evals) - 1)) != 0:
        raise ValueError("len(evals) must be a power of two")
    ss = int(shift) % FR_MODULUS
    inv_s = fr_inv(ss) if ss != 0 else 0
    out = [int(x) % FR_MODULUS for x in evals]
    intt_inplace(out, omega)
    pow_inv_s = 1
    for i in range(len(out)):
        out[i] = out[i] * pow_inv_s % FR_MODULUS
        pow_inv_s = (pow_inv_s * inv_s) % FR_MODULUS if inv_s != 0 else 0
    return out

# 利用 NTT 实现多项式乘法
def poly_mul_ntt(a: Sequence[int], b: Sequence[int]) -> List[int]:
    if len(a) == 0 or len(b) == 0:
        return []
    need = len(a) + len(b) - 1
    n = 1 << (need - 1).bit_length()
    omega = omega_for_size(n)
    a_eval = evals_from_coeffs_on_roots(a, n=n, omega=omega)
    b_eval = evals_from_coeffs_on_roots(b, n=n, omega=omega)
    c_eval = [(a_eval[i] * b_eval[i]) % FR_MODULUS for i in range(n)]
    c_coeff = coeffs_from_evals_on_roots(c_eval, omega=omega)
    out = c_coeff[:need]
    while len(out) > 0 and out[-1] % FR_MODULUS == 0:
        out.pop()
    return out
