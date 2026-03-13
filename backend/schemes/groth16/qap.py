from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.poly import lagrange_interpolate, poly_divmod, poly_mul, poly_sub, poly_vanishing_from_roots


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
