import secrets
import unittest

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.poly import (
    coeffs_from_evals_on_roots,
    evals_from_coeffs_on_coset,
    evals_from_coeffs_on_roots,
    intt,
    ntt,
    omega_for_size,
    poly_mul,
    poly_mul_ntt,
    poly_div_by_xn_minus_1,
    roots_of_unity,
)
from pyZKP.common.crypto.poly.cpu_ref import poly_eval
from pyZKP.common.crypto.poly.cpu_ref import poly_divmod


class TestPolyNTT(unittest.TestCase):
    def test_ntt_roundtrip(self):
        for n in [2, 4, 8, 16, 32]:
            omega = omega_for_size(n)
            coeffs = [secrets.randbelow(FR_MODULUS) for _ in range(n)]
            evals = evals_from_coeffs_on_roots(coeffs, n=n, omega=omega)
            back = coeffs_from_evals_on_roots(evals, omega=omega)
            self.assertEqual([c % FR_MODULUS for c in coeffs], [c % FR_MODULUS for c in back])

    def test_ntt_matches_basic_api(self):
        for n in [2, 4, 8, 16]:
            omega = omega_for_size(n)
            coeffs = [secrets.randbelow(FR_MODULUS) for _ in range(n)]
            evals1 = evals_from_coeffs_on_roots(coeffs, n=n, omega=omega)
            evals2 = ntt(coeffs, omega=omega)
            self.assertEqual([x % FR_MODULUS for x in evals1], [x % FR_MODULUS for x in evals2])

            back1 = coeffs_from_evals_on_roots(evals1, omega=omega)
            back2 = intt(evals2, omega=omega)
            self.assertEqual([x % FR_MODULUS for x in back1], [x % FR_MODULUS for x in back2])

    def test_poly_mul_via_ntt(self):
        for n in [8, 16, 32]:
            omega = omega_for_size(n)
            a = [secrets.randbelow(FR_MODULUS) for _ in range(n // 4)]
            b = [secrets.randbelow(FR_MODULUS) for _ in range(n // 4)]
            prod_ref = poly_mul(a, b)

            a_eval = evals_from_coeffs_on_roots(a, n=n, omega=omega)
            b_eval = evals_from_coeffs_on_roots(b, n=n, omega=omega)
            c_eval = [(a_eval[i] * b_eval[i]) % FR_MODULUS for i in range(n)]
            c_coeff = coeffs_from_evals_on_roots(c_eval, omega=omega)

            self.assertEqual([x % FR_MODULUS for x in prod_ref], [x % FR_MODULUS for x in c_coeff[: len(prod_ref)]])

    def test_poly_mul_ntt_matches_reference(self):
        for _ in range(20):
            la = secrets.randbelow(20) + 1
            lb = secrets.randbelow(20) + 1
            a = [secrets.randbelow(FR_MODULUS) for _ in range(la)]
            b = [secrets.randbelow(FR_MODULUS) for _ in range(lb)]
            self.assertEqual(poly_mul(a, b), poly_mul_ntt(a, b))

    def test_coset_evals_match_direct_eval(self):
        for n in [8, 16, 32]:
            omega = omega_for_size(n)
            roots = roots_of_unity(n, omega)
            shift = 5
            coeffs = [secrets.randbelow(FR_MODULUS) for _ in range(n // 4)]
            evals = evals_from_coeffs_on_coset(coeffs, n=n, omega=omega, shift=shift)
            for i in range(n):
                x = (shift * roots[i]) % FR_MODULUS
                self.assertEqual(evals[i] % FR_MODULUS, poly_eval(coeffs, x) % FR_MODULUS)

    def test_div_by_xn_minus_1_matches_divmod(self):
        for n in [8, 16, 32]:
            for _ in range(20):
                deg = secrets.randbelow(3 * n) + 1
                num = [secrets.randbelow(FR_MODULUS) for _ in range(deg)]
                den = [(-1) % FR_MODULUS] + [0] * (n - 1) + [1]
                q1, r1 = poly_div_by_xn_minus_1(num, n)
                q2, r2 = poly_divmod(num, den)
                self.assertEqual([x % FR_MODULUS for x in q1], [x % FR_MODULUS for x in q2])
                self.assertEqual([x % FR_MODULUS for x in r1], [x % FR_MODULUS for x in r2])


if __name__ == "__main__":
    unittest.main()
