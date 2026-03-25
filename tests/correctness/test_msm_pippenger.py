import secrets
import unittest

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.kzg.cpu_ref import setup_srs
from pyZKP.common.crypto.ecc.bn254 import g1_eq, g2_eq
from pyZKP.common.crypto.msm import fixed_base_precompute, msm_fixed_base, msm_fixed_base_batch, msm_naive_g1, msm_naive_g2, msm_pippenger, msm_pippenger_batch, msm_pippenger_g2


class TestMSMPippenger(unittest.TestCase):
    def test_pippenger_matches_naive_small(self):
        srs = setup_srs(256)
        points = list(srs.g1_powers[:64])
        scalars = [secrets.randbelow(FR_MODULUS) for _ in range(64)]

        a = msm_naive_g1(points, scalars)
        b = msm_pippenger(points, scalars, window_bits=8)
        self.assertTrue(g1_eq(a, b))

    def test_pippenger_matches_naive_random_sizes(self):
        srs = setup_srs(512)
        for n in [1, 2, 3, 8, 31, 32, 63, 64, 127]:
            points = list(srs.g1_powers[:n])
            scalars = [secrets.randbelow(FR_MODULUS) for _ in range(n)]
            a = msm_naive_g1(points, scalars)
            b = msm_pippenger(points, scalars, window_bits=7)
            self.assertTrue(g1_eq(a, b))

    def test_pippenger_batch_matches_single(self):
        srs = setup_srs(512)
        points = list(srs.g1_powers[:256])
        scalars_list = []
        for n in [200, 128, 1, 0]:
            scalars_list.append([secrets.randbelow(FR_MODULUS) for _ in range(n)])
        outs = msm_pippenger_batch(points, scalars_list, window_bits=16)
        expected = [msm_pippenger(points[: len(s)], s, window_bits=16) for s in scalars_list]
        self.assertEqual(len(outs), len(expected))
        for a, b in zip(outs, expected):
            self.assertTrue(g1_eq(a, b))

    def test_fixed_base_matches_pippenger(self):
        srs = setup_srs(256)
        points = tuple(srs.g1_powers[:64])
        scalars = [secrets.randbelow(FR_MODULUS) for _ in range(64)]
        pre = fixed_base_precompute(points, 8)
        a = msm_fixed_base(pre, scalars)
        b = msm_pippenger(points, scalars, window_bits=8)
        self.assertTrue(g1_eq(a, b))

    def test_fixed_base_batch_matches_pippenger_batch(self):
        srs = setup_srs(256)
        points = tuple(srs.g1_powers[:64])
        scalars_list = []
        for _ in range(4):
            scalars_list.append([secrets.randbelow(FR_MODULUS) for _ in range(64)])
        pre = fixed_base_precompute(points, 8)
        a = msm_fixed_base_batch(pre, scalars_list)
        b = msm_pippenger_batch(points, scalars_list, window_bits=8)
        for x, y in zip(a, b):
            self.assertTrue(g1_eq(x, y))

    def test_pippenger_g2_matches_naive_small(self):
        srs = setup_srs(256)
        points = list(srs.g2_powers[:64])
        scalars = [secrets.randbelow(FR_MODULUS) for _ in range(64)]
        a = msm_naive_g2(points, scalars)
        b = msm_pippenger_g2(points, scalars, window_bits=8)
        self.assertTrue(g2_eq(a, b))

    def test_pippenger_g2_matches_naive_random_sizes(self):
        srs = setup_srs(256)
        for n in [1, 2, 3, 8, 31, 32, 63, 64]:
            points = list(srs.g2_powers[:n])
            scalars = [secrets.randbelow(FR_MODULUS) for _ in range(n)]
            a = msm_naive_g2(points, scalars)
            b = msm_pippenger_g2(points, scalars, window_bits=7)
            self.assertTrue(g2_eq(a, b))


if __name__ == "__main__":
    unittest.main()
