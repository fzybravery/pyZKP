import unittest
from frontend.api.api import API, Circuit, Var
from frontend.api.compile import compile_circuit
from frontend.api.witness import build_witness, check_r1cs
from crypto.field.fr import FR_MODULUS, fr_inv
from frontend.circuit.schema import public, secret

# 测试 And, Or, Xor, Neg 算子
class LogicOpsCircuit(Circuit):
    def __init__(self):
        self.a = secret()
        self.b = secret()
        self.expected_and = public()
        self.expected_or = public()
        self.expected_xor = public()
        self.val = secret()
        self.expected_neg = public()

    def define(self, api: API) -> None:
        # Boolean checks will be added by the API for logical ops inputs
        and_res = api.And(self.a, self.b)
        api.AssertIsEqual(and_res, self.expected_and)

        or_res = api.Or(self.a, self.b)
        api.AssertIsEqual(or_res, self.expected_or)

        xor_res = api.Xor(self.a, self.b)
        api.AssertIsEqual(xor_res, self.expected_xor)

        neg_res = api.Neg(self.val)
        api.AssertIsEqual(neg_res, self.expected_neg)

# 测试 Inverse, Div, AssertIsDifferent 算子
class FieldOpsCircuit(Circuit):
    def __init__(self):
        self.x = secret()
        self.y = secret()
        self.expected_inv_x = public()
        self.expected_div_x_y = public()

    def define(self, api: API) -> None:
        api.AssertIsDifferent(self.x, api.Constant(0)) # Ensure x != 0
        api.AssertIsDifferent(self.y, api.Constant(0)) # Ensure y != 0
        api.AssertIsDifferent(self.x, self.y) # Ensure x != y
        
        inv_x = api.Inverse(self.x)
        api.AssertIsEqual(inv_x, self.expected_inv_x)

        div_res = api.Div(self.x, self.y)
        api.AssertIsEqual(div_res, self.expected_div_x_y)

# 测试 ToBinary, FromBinary 算子
class BinaryOpsCircuit(Circuit):
    def __init__(self):
        self.val = secret()
        self.expected_b0 = public()
        self.expected_b1 = public()
        self.expected_b2 = public()

    def define(self, api: API) -> None:
        bits = api.ToBinary(self.val, 3)
        api.AssertIsEqual(bits[0], self.expected_b0)
        api.AssertIsEqual(bits[1], self.expected_b1)
        api.AssertIsEqual(bits[2], self.expected_b2)

        recomposed = api.FromBinary(bits)
        api.AssertIsEqual(recomposed, self.val)

class TestFrontendAdvancedOps(unittest.TestCase):
    def test_logic_ops(self):
        circuit = LogicOpsCircuit()
        ir = compile_circuit(circuit, FR_MODULUS)
        
        # Test 1 & 0
        assignment = {
            "a": 1,
            "b": 0,
            "expected_and": 0,
            "expected_or": 1,
            "expected_xor": 1,
            "val": 5,
            "expected_neg": (-5) % FR_MODULUS
        }
        witness = build_witness(ir, assignment)
        check_r1cs(ir, witness)

        # Test 1 & 1
        assignment2 = {
            "a": 1,
            "b": 1,
            "expected_and": 1,
            "expected_or": 1,
            "expected_xor": 0,
            "val": 0,
            "expected_neg": 0
        }
        witness2 = build_witness(ir, assignment2)
        check_r1cs(ir, witness2)

        # Invalid Boolean Input
        assignment_invalid = {
            "a": 2, # Not a boolean!
            "b": 0,
            "expected_and": 0,
            "expected_or": 1,
            "expected_xor": 1,
            "val": 5,
            "expected_neg": (-5) % FR_MODULUS
        }
        with self.assertRaises(AssertionError):
            # AssertIsBoolean should fail
            witness_invalid = build_witness(ir, assignment_invalid)
            check_r1cs(ir, witness_invalid)

    def test_field_ops(self):
        circuit = FieldOpsCircuit()
        ir = compile_circuit(circuit, FR_MODULUS)
        
        x = 10
        y = 3
        inv_x = fr_inv(x)
        inv_y = fr_inv(y)
        div_x_y = (x * inv_y) % FR_MODULUS

        assignment = {
            "x": x,
            "y": y,
            "expected_inv_x": inv_x,
            "expected_div_x_y": div_x_y
        }
        witness = build_witness(ir, assignment)
        check_r1cs(ir, witness)

        # Test AssertIsDifferent failure (x == y)
        assignment_same = {
            "x": x,
            "y": x,
            "expected_inv_x": inv_x,
            "expected_div_x_y": 1
        }
        witness_same = build_witness(ir, assignment_same)
        with self.assertRaises(AssertionError):
            check_r1cs(ir, witness_same)

        # Test AssertIsDifferent failure (x == 0)
        assignment_zero = {
            "x": 0,
            "y": y,
            "expected_inv_x": 0, # irrelevant, should fail earlier
            "expected_div_x_y": 0
        }
        witness_zero = build_witness(ir, assignment_zero)
        with self.assertRaises(AssertionError):
            check_r1cs(ir, witness_zero)

    def test_binary_ops(self):
        circuit = BinaryOpsCircuit()
        ir = compile_circuit(circuit, FR_MODULUS)

        # 6 in binary is 110 (b0=0, b1=1, b2=1)
        assignment = {
            "val": 6,
            "expected_b0": 0,
            "expected_b1": 1,
            "expected_b2": 1
        }
        witness = build_witness(ir, assignment)
        check_r1cs(ir, witness)

        # Try to decompose a number that requires more than 3 bits (e.g. 8 = 1000)
        assignment_overflow = {
            "val": 8,
            "expected_b0": 0,
            "expected_b1": 0,
            "expected_b2": 0
        }
        with self.assertRaises(AssertionError):
            # Recomposed value will be 0, but val is 8. assert_is_equal(x, recomposed) will fail.
            witness_overflow = build_witness(ir, assignment_overflow)
            check_r1cs(ir, witness_overflow)

if __name__ == '__main__':
    unittest.main()
