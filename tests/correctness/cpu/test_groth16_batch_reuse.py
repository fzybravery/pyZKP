import unittest
from dataclasses import dataclass

from frontend.api import build_witness, check_r1cs, compile_circuit
from backend.schemes.groth16 import setup, verify
from backend.schemes.groth16.prove import prove_batch
from frontend.circuit.schema import public, secret


BN254_FR_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


@dataclass
class CubicCircuit:
    x: object = secret("x")
    y: object = public("y")

    def define(self, api):
        x3 = api.Mul(self.x, self.x, self.x)
        api.AssertIsEqual(self.y, api.Add(x3, self.x, 5))


class TestGroth16BatchReuse(unittest.TestCase):
    def test_prove_batch_outputs_verifiable_proofs(self):
        ir = compile_circuit(CubicCircuit(), BN254_FR_MODULUS)
        pk = setup(ir)

        xs = [9, 10, 11]
        witnesses = []
        publics = []
        for x in xs:
            y = (x * x * x + x + 5) % BN254_FR_MODULUS
            wit = build_witness(ir, {"x": x, "y": y})
            check_r1cs(ir, wit)
            witnesses.append(wit)
            publics.append([1, y])

        proofs = prove_batch(ir, pk, witnesses)
        self.assertEqual(len(proofs), len(witnesses))
        for p, pub in zip(proofs, publics):
            self.assertTrue(verify(pk.vk, pub, p))


if __name__ == "__main__":
    unittest.main()

