from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.poly import (
    coeffs_from_evals_on_roots,
    poly_div_by_xn_minus_1,
    poly_mul_ntt,
    poly_sub,
)


@dataclass(frozen=True)
class QAPWitnessPolys:
    a_poly: List[int]
    b_poly: List[int]
    c_poly: List[int]
    h_poly: List[int]
    t_poly: List[int]


def compute_h_from_abc(xs: Sequence[int], a_eval: Sequence[int], b_eval: Sequence[int], c_eval: Sequence[int]) -> QAPWitnessPolys:
    if not (len(xs) == len(a_eval) == len(b_eval) == len(c_eval)):
        raise ValueError("length mismatch")
    a_poly = lagrange_interpolate(xs, list(a_eval))
    b_poly = lagrange_interpolate(xs, list(b_eval))
    c_poly = lagrange_interpolate(xs, list(c_eval))
    t_poly = poly_vanishing_from_roots(xs)
    p_poly = poly_sub(poly_mul(a_poly, b_poly), c_poly)
    q, r = poly_divmod(p_poly, t_poly)
    if len(r) != 0:
        raise ValueError("witness does not satisfy R1CS (non-zero remainder)")
    return QAPWitnessPolys(a_poly=a_poly, b_poly=b_poly, c_poly=c_poly, h_poly=q, t_poly=t_poly)


def compute_h_from_abc_on_roots(n: int, omega: int, a_eval: Sequence[int], b_eval: Sequence[int], c_eval: Sequence[int]) -> QAPWitnessPolys:
    if not (len(a_eval) == len(b_eval) == len(c_eval) == n):
        raise ValueError("length mismatch")
    a_poly = coeffs_from_evals_on_roots(a_eval, omega=omega)
    b_poly = coeffs_from_evals_on_roots(b_eval, omega=omega)
    c_poly = coeffs_from_evals_on_roots(c_eval, omega=omega)
    t_poly = [(-1) % FR_MODULUS] + [0] * (n - 1) + [1]
    p_poly = poly_sub(poly_mul_ntt(a_poly, b_poly), list(c_poly))
    q, r = poly_div_by_xn_minus_1(p_poly, n)
    if len(r) != 0:
        raise ValueError("witness does not satisfy R1CS (non-zero remainder)")
    return QAPWitnessPolys(a_poly=list(a_poly), b_poly=list(b_poly), c_poly=list(c_poly), h_poly=q, t_poly=t_poly)
