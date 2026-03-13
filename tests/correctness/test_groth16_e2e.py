import unittest
from dataclasses import dataclass

from pyZKP import build_witness, check_r1cs, compile_circuit
from pyZKP.backend.schemes.groth16 import prove, setup, verify


BN254_FR_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


@dataclass
class CubicCircuit:
    x: object
    y: object

    def __init__(self):
        from pyZKP.frontend.circuit.schema import public, secret

        self.x = secret("x")
        self.y = public("y")

    def define(self, api):
        x3 = api.Mul(self.x, self.x, self.x)
        api.AssertIsEqual(self.y, api.Add(x3, self.x, 5))


class TestGroth16E2E(unittest.TestCase):
    def test_groth16_end_to_end(self):
        ir = compile_circuit(CubicCircuit(), BN254_FR_MODULUS)
        x = 9
        y = (x * x * x + x + 5) % BN254_FR_MODULUS
        wit = build_witness(ir, {"x": x, "y": y})
        check_r1cs(ir, wit)

        pk = setup(ir)
        prf = prove(ir, pk, wit)
        ok = verify(pk.vk, [1, y], prf)
        self.assertTrue(ok)

    def test_groth16_wrong_public_fails(self):
        ir = compile_circuit(CubicCircuit(), BN254_FR_MODULUS)
        x = 9
        y = (x * x * x + x + 5) % BN254_FR_MODULUS
        wit = build_witness(ir, {"x": x, "y": y})
        pk = setup(ir)
        prf = prove(ir, pk, wit)
        ok = verify(pk.vk, [1, (y + 1) % BN254_FR_MODULUS], prf)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
