import secrets
import unittest

from pyZKP.common.crypto.field import FR_MODULUS, fr_batch_inv, fr_inv


class TestFieldBatchInv(unittest.TestCase):
    def test_batch_inv_matches_single_inv(self):
        xs = [secrets.randbelow(FR_MODULUS) for _ in range(200)]
        xs[0] = 0
        xs[5] = 0
        invs = fr_batch_inv(xs)
        self.assertEqual(len(invs), len(xs))
        for x, invx in zip(xs, invs):
            xx = int(x) % FR_MODULUS
            if xx == 0:
                self.assertEqual(invx, 0)
            else:
                self.assertEqual(invx % FR_MODULUS, fr_inv(xx) % FR_MODULUS)
                self.assertEqual((xx * invx) % FR_MODULUS, 1)


if __name__ == "__main__":
    unittest.main()

