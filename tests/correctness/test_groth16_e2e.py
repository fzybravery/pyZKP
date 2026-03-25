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



"""
Groth16 零知识证明系统的全链路集成测试模块。

测试用例基于 CubicCircuit (y = x^3 + x + 5) 构建。

核心测试点：
1. test_groth16_end_to_end: 
   正向闭环测试。验证电路能正确编译为 IR，基于正确的 Witness 能够生成合法的 Groth16 证明 (A, B, C)，
   且验证者 (Verifier) 利用验证钥 (VK) 和公开输入能通过双线性配对校验。
2. test_groth16_wrong_public_fails: 
   反向安全性测试。验证协议的 Soundness（可靠性）。当验证者侧提供的公开输入（Public Inputs）
   与证明者生成证明时使用的输入不一致时，配对校验必须失败。
"""

class TestGroth16E2E(unittest.TestCase):
    def test_groth16_end_to_end(self):
        ir = compile_circuit(CubicCircuit(), BN254_FR_MODULUS)
        x = 9
        y = (x * x * x + x + 5) % BN254_FR_MODULUS
        wit = build_witness(ir, {"x": x, "y": y})
        check_r1cs(ir, wit)

        pk = setup(ir)
        prf = prove(ir, pk, wit, runtime_attrs={"fixed_base_policy": "auto", "fixed_base_auto_groth16_min_calls": 1})
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
