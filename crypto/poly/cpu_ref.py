from __future__ import annotations

from typing import List, Sequence, Tuple

from crypto.field.fr import FR_MODULUS, fr_inv


def poly_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        out[i] = (av + bv) % FR_MODULUS
    return _trim(out)


def poly_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        out[i] = (av - bv) % FR_MODULUS
    return _trim(out)


def poly_scale(a: Sequence[int], k: int) -> List[int]:
    kk = k % FR_MODULUS
    return _trim([(x * kk) % FR_MODULUS for x in a])


def poly_mul(a: Sequence[int], b: Sequence[int]) -> List[int]:
    if len(a) == 0 or len(b) == 0:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        if av == 0:
            continue
        for j, bv in enumerate(b):
            out[i + j] = (out[i + j] + av * bv) % FR_MODULUS
    return _trim(out)


def poly_eval(a: Sequence[int], x: int) -> int:
    xx = x % FR_MODULUS
    acc = 0
    for c in reversed(a):
        acc = (acc * xx + (c % FR_MODULUS)) % FR_MODULUS
    return acc


def poly_divmod(num: Sequence[int], den: Sequence[int]) -> Tuple[List[int], List[int]]:
    n = _trim(list(num))
    d = _trim(list(den))
    if len(d) == 0:
        raise ZeroDivisionError
    if len(n) < len(d):
        return [], n
    inv_lead = fr_inv(d[-1])
    q = [0] * (len(n) - len(d) + 1)
    r = n[:]
    for k in range(len(q) - 1, -1, -1):
        coeff = (r[len(d) - 1 + k] * inv_lead) % FR_MODULUS
        q[k] = coeff
        if coeff != 0:
            for j in range(len(d)):
                r[j + k] = (r[j + k] - coeff * d[j]) % FR_MODULUS
    return _trim(q), _trim(r)


def poly_vanishing_from_roots(xs: Sequence[int]) -> List[int]:
    t = [1]
    for x in xs:
        t = poly_mul(t, [(-x) % FR_MODULUS, 1])
    return t


def lagrange_interpolate(xs: Sequence[int], ys: Sequence[int]) -> List[int]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys length mismatch")
    n = len(xs)
    out = [0]
    for i in range(n):
        num = [1]
        den = 1
        xi = xs[i] % FR_MODULUS
        for j in range(n):
            if i == j:
                continue
            xj = xs[j] % FR_MODULUS
            num = poly_mul(num, [(-xj) % FR_MODULUS, 1])
            den = (den * (xi - xj)) % FR_MODULUS
        li = poly_scale(num, fr_inv(den))
        out = poly_add(out, poly_scale(li, ys[i]))
    return _trim(out)


def barycentric_precompute(xs: Sequence[int]) -> Tuple[List[int], List[int]]:
    x = [v % FR_MODULUS for v in xs]
    w: List[int] = []
    for i in range(len(x)):
        acc = 1
        xi = x[i]
        for j in range(len(x)):
            if i == j:
                continue
            acc = (acc * (xi - x[j])) % FR_MODULUS
        w.append(fr_inv(acc))
    return x, w


def barycentric_value(xs: Sequence[int], ws: Sequence[int], ys: Sequence[int], at: int) -> int:
    if len(xs) != len(ws) or len(xs) != len(ys):
        raise ValueError("length mismatch")
    x = [v % FR_MODULUS for v in xs]
    atv = at % FR_MODULUS
    for xi, yi in zip(x, ys):
        if atv == xi:
            return yi % FR_MODULUS
    num = 0
    den = 0
    for xi, wi, yi in zip(x, ws, ys):
        inv = fr_inv((atv - xi) % FR_MODULUS)
        t = (wi * inv) % FR_MODULUS
        num = (num + t * (yi % FR_MODULUS)) % FR_MODULUS
        den = (den + t) % FR_MODULUS
    return (num * fr_inv(den)) % FR_MODULUS


def _trim(a: List[int]) -> List[int]:
    while len(a) > 0 and a[-1] % FR_MODULUS == 0:
        a.pop()
    return a
