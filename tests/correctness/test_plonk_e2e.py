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


"""
PLONK 零知识证明系统的全链路集成测试模块。

测试用例基于 CubicCircuit (y = x^3 + x + 5) 算术电路构建。

核心测试点：
1. test_plonk_e2e: 
   正向闭环测试。验证系统能够正确进行电路降级与预处理 (Setup)。
   证明者 (Prover) 能够基于正确的 Witness 和公共输入生成合法的 PLONK 证明；
   验证者 (Verifier) 利用验证钥 (VK)、公开输入及证明，通过多项式求值和 KZG 配对校验。
2. test_plonk_wrong_public_fails: 
   反向安全性 (Soundness) 测试。模拟公开输入数据不一致的场景。
   当验证者侧提供的公共输入被篡改 (如 y+1) 时，因公开多项式 PI(X) 改变，
   必然导致最终的商多项式恒等式验证失败，确保系统具备抗伪造性。
"""

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
