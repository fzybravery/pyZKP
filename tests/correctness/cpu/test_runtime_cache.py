import unittest

from common.crypto.field.fr import FR_MODULUS
from runtime.cache import circuit_ir_fingerprint
from runtime.bench import RepeatMulCircuit
from frontend.api import compile_circuit


class TestRuntimeCache(unittest.TestCase):
    def test_circuit_ir_fingerprint_stable(self):
        ir1 = compile_circuit(RepeatMulCircuit(10), FR_MODULUS)
        ir2 = compile_circuit(RepeatMulCircuit(10), FR_MODULUS)
        self.assertEqual(circuit_ir_fingerprint(ir1), circuit_ir_fingerprint(ir2))

    def test_circuit_ir_fingerprint_changes(self):
        ir1 = compile_circuit(RepeatMulCircuit(10), FR_MODULUS)
        ir2 = compile_circuit(RepeatMulCircuit(11), FR_MODULUS)
        self.assertNotEqual(circuit_ir_fingerprint(ir1), circuit_ir_fingerprint(ir2))


if __name__ == "__main__":
    unittest.main()

