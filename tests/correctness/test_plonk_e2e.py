import unittest
from dataclasses import dataclass

from pyZKP import build_witness, check_r1cs, compile_circuit
from pyZKP.backend.schemes.plonk import prove, setup, verify
from pyZKP.frontend.circuit.schema import public, secret


BN254_FR_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


@dataclass
class CubicCircuit:
    x: object = secret("x")
    y: object = public("y")

    def define(self, api):
        x3 = api.Mul(self.x, self.x, self.x)
        api.AssertIsEqual(self.y, api.Add(x3, self.x, 5))


class TestPlonkE2E(unittest.TestCase):
    def test_plonk_end_to_end(self):
        ir = compile_circuit(CubicCircuit(), BN254_FR_MODULUS)
        x = 9
        y = (x * x * x + x + 5) % BN254_FR_MODULUS
        wit = build_witness(ir, {"x": x, "y": y})
        check_r1cs(ir, wit)

        pk = setup(ir)
        prf = prove(pk, wit, public_values=[1, y])
        ok = verify(pk.vk, prf, public_values=[1, y])
        self.assertTrue(ok)

    def test_plonk_wrong_public_fails(self):
        ir = compile_circuit(CubicCircuit(), BN254_FR_MODULUS)
        x = 9
        y = (x * x * x + x + 5) % BN254_FR_MODULUS
        wit = build_witness(ir, {"x": x, "y": y})
        pk = setup(ir)
        prf = prove(pk, wit, public_values=[1, y])
        ok = verify(pk.vk, prf, public_values=[1, (y + 1) % BN254_FR_MODULUS])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
