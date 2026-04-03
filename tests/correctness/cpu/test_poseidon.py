import unittest
from frontend.api.api import API, Circuit
from frontend.api.compile import compile_circuit
from frontend.api.witness import build_witness, check_r1cs
from frontend.circuit.schema import public, secret
from crypto.field.fr import FR_MODULUS
from crypto.hash.poseidon import poseidon_hash
from frontend.api.std.poseidon import poseidon_circuit

class PoseidonCircuit(Circuit):
    def __init__(self):
        self.in0 = secret()
        self.in1 = secret()
        self.expected = public()

    def define(self, api: API) -> None:
        # 使用 Poseidon 电路计算两个输入的哈希
        out = poseidon_circuit(api, [self.in0, self.in1])
        # 断言电路计算出的哈希等于外部提供的 expected 值
        api.AssertIsEqual(out, self.expected)

class TestPoseidon(unittest.TestCase):
    def test_poseidon_hash_circuit_correctness(self):
        """
        验证纯 Python 的 Poseidon Hash 与电路版本生成的约束是否完全一致。
        """
        in0 = 12345
        in1 = 67890
        
        # 1. 外部用纯 Python 逻辑计算出哈希期望值
        expected_hash = poseidon_hash([in0, in1])
        
        # 2. 编译电路生成 R1CS 约束系统
        circuit = PoseidonCircuit()
        ir = compile_circuit(circuit, FR_MODULUS)
        
        # 3. 构造 Witness (赋值)
        assignment = {
            "in0": in0,
            "in1": in1,
            "expected": expected_hash
        }
        witness = build_witness(ir, assignment)
        
        # 4. 验证 R1CS (如果正确，check_r1cs 不会抛出 AssertionError)
        check_r1cs(ir, witness)
        
        # 打印生成的约束数量以展示 Poseidon 在 ZKP 中的极高效率
        print(f"\n[Poseidon] t=3 (2 inputs), R_f=8, R_p=53. Total R1CS Constraints: {len(ir.constraints)}")
        
        # 5. 反面测试：提供错误的期望值，应当验证失败
        bad_assignment = {
            "in0": in0,
            "in1": in1,
            "expected": expected_hash + 1
        }
        with self.assertRaises(AssertionError):
            bad_witness = build_witness(ir, bad_assignment)
            check_r1cs(ir, bad_witness)

if __name__ == '__main__':
    unittest.main()
