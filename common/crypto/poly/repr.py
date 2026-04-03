from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from common.crypto.field.fr import FR_MODULUS
from common.crypto.poly.cpu_ref import poly_eval
from common.crypto.poly.ntt import coeffs_from_evals_on_roots, evals_from_coeffs_on_roots

"""
当前文件将多项式封装为 PolyCoeffs 和 PolyEvals 两个不可变对象
"""

# 多项式系数表示
@dataclass(frozen=True)
class PolyCoeffs:
    coeffs: Tuple[int, ...]

    def value(self, x: int) -> int:
        return poly_eval(self.coeffs, x) % FR_MODULUS

    def to_evals(self, *, n: int, omega: int) -> "PolyEvals":
        ev = evals_from_coeffs_on_roots(self.coeffs, n=n, omega=omega)
        return PolyEvals(evals=tuple(ev), n=n, omega=int(omega) % FR_MODULUS)

# 多项式点值表示
@dataclass(frozen=True)
class PolyEvals:
    evals: Tuple[int, ...]
    n: int
    omega: int

    def to_coeffs(self) -> PolyCoeffs:
        coeffs = coeffs_from_evals_on_roots(self.evals, omega=self.omega)
        return PolyCoeffs(coeffs=tuple(coeffs))

