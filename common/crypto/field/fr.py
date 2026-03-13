from __future__ import annotations

import secrets

from py_ecc import optimized_bn128 as b

FR_MODULUS = int(b.curve_order)


def fr_add(a: int, c: int) -> int:
    return (a + c) % FR_MODULUS


def fr_sub(a: int, c: int) -> int:
    return (a - c) % FR_MODULUS


def fr_neg(a: int) -> int:
    return (-a) % FR_MODULUS


def fr_mul(a: int, c: int) -> int:
    return (a * c) % FR_MODULUS


def fr_pow(a: int, e: int) -> int:
    return pow(a % FR_MODULUS, e, FR_MODULUS)


def fr_inv(a: int) -> int:
    aa = a % FR_MODULUS
    if aa == 0:
        raise ZeroDivisionError
    return pow(aa, FR_MODULUS - 2, FR_MODULUS)


def fr_rand(nonzero: bool = True) -> int:
    while True:
        x = secrets.randbelow(FR_MODULUS)
        if nonzero and x == 0:
            continue
        return x
