import unittest
from dataclasses import dataclass

from frontend.api import build_witness, check_r1cs, compile_circuit
from frontend.ir.core import LinExpr, VarRef, Visibility
from frontend.circuit.schema import public, secret


BN254_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# 立方电路
@dataclass
class CubicCircuit:
    x: object = secret("x")
    y: object = public("y")

    def define(self, api):
        x3 = api.Mul(self.x, self.x, self.x)
        api.AssertIsEqual(self.y, api.Add(x3, self.x, 5))

# 布尔选择电路
@dataclass
class BoolSelectCircuit:
    a: object = public("a")
    b: object = public("b")
    flag: object = secret("flag")
    out: object = public("out")

    def define(self, api):
        z = api.IsZero(api.Sub(self.a, self.b))
        sel = api.Select(self.flag, self.a, self.b)
        api.AssertIsEqual(self.out, api.Add(sel, z))

# 同名电路
@dataclass
class DuplicateNameCircuit:
    a: object = public("dup")
    b: object = secret("dup")

    def define(self, api):
        api.AssertIsEqual(self.a, self.b)

# 二进制转换电路
@dataclass
class ToBinaryCircuit:
    x: object = secret("x")
    out: object = public("out")

    def define(self, api):
        bits = api.ToBinary(self.x, 8)
        recomposed = api.FromBinary(bits)
        api.AssertIsEqual(self.out, api.Add(recomposed, 0))
        api.AssertIsEqual(self.x, recomposed)


def double_hint(x: int) -> int:
    return x * 2

# hint电路
@dataclass
class UserHintCircuit:
    x: object = public("x")
    out: object = public("out")

    def define(self, api):
        y = api.Hint(double_hint, [self.x], n_outputs=1, names=["double"])
        api.AssertIsEqual(self.out, y)
        api.AssertIsEqual(y, api.Mul(2, self.x))


"""
pyZKP 前端编译与 Witness 生成核心测试套件。

覆盖的核心特性包括：
1. Schema 解析与排序：确保 Public 变量始终分配在 Secret 变量之前，且能拦截同名输入错误。
2. IR 降级逻辑：验证 api.Mul 等非线性操作被正确拆解为 R1CS 约束，并生成对应的内部中间变量。
3. 复杂组件编译：验证 api.IsZero, api.Select 以及 api.ToBinary 等高阶组件的约束生成正确性。
4. 内存引用安全：检查生成的所有 R1CS 约束矩阵中引用的变量 ID 均在合法范围内。
5. 端到端求解与校验：利用 build_witness 基于初始输入推导所有内部变量（包含执行 Python 原生的 Hint 函数），
   并利用 check_r1cs 严格验证这些推导值是否满足所有多项式约束。
"""

class TestFrontendCompileIR(unittest.TestCase):
    def test_public_secret_input_order(self):
        ir = compile_circuit(CubicCircuit(), BN254_MODULUS)
        print(ir)
        self.assertEqual([i.name for i in ir.inputs], ["y", "x"])
        self.assertEqual([i.visibility for i in ir.inputs], [Visibility.PUBLIC, Visibility.SECRET])
        self.assertEqual(ir.vars[0].name, "y")
        self.assertEqual(ir.vars[1].name, "x")

    def test_mul_introduces_internal_vars_and_r1cs_constraints(self):
        ir = compile_circuit(CubicCircuit(), BN254_MODULUS)
        internal = [v for v in ir.vars if v.visibility == Visibility.INTERNAL]
        self.assertGreaterEqual(len(internal), 2)
        self.assertGreaterEqual(len(ir.constraints), 3)
        for c in ir.constraints:
            self.assertIsInstance(c.a, LinExpr)
            self.assertIsInstance(c.b, LinExpr)
            self.assertIsInstance(c.c, LinExpr)

    def test_iszero_select_compiles(self):
        ir = compile_circuit(BoolSelectCircuit(), BN254_MODULUS)
        self.assertGreaterEqual(len(ir.constraints), 1)
        self.assertGreaterEqual(len(ir.vars), 1)

    def test_schema_rejects_duplicate_input_names(self):
        with self.assertRaises(ValueError):
            compile_circuit(DuplicateNameCircuit(), BN254_MODULUS)

    def test_all_varrefs_in_constraints_exist(self):
        ir = compile_circuit(BoolSelectCircuit(), BN254_MODULUS)
        max_id = max(v.id for v in ir.vars)

        def check_lin(le: LinExpr):
            for vid, _ in le.terms:
                self.assertLessEqual(vid, max_id)

        for c in ir.constraints:
            check_lin(c.a)
            check_lin(c.b)
            check_lin(c.c)

    def test_build_witness_and_check_constraints(self):
        ir = compile_circuit(BoolSelectCircuit(), BN254_MODULUS)
        wit = build_witness(ir, {"a": 7, "b": 7, "flag": 1, "out": 8})
        print(f"wit: {wit}")
        check_r1cs(ir, wit)

        wit2 = build_witness(ir, {"a": 7, "b": 9, "flag": 0, "out": 9})
        print(f"wit2: {wit2}")
        check_r1cs(ir, wit2)

    def test_to_binary_hint_and_witness(self):
        ir = compile_circuit(ToBinaryCircuit(), BN254_MODULUS)
        wit = build_witness(ir, {"x": 13, "out": 13})
        check_r1cs(ir, wit)

    def test_user_hint_registration_and_witness(self):
        ir = compile_circuit(UserHintCircuit(), BN254_MODULUS)
        wit = build_witness(ir, {"x": 9, "out": 18})
        check_r1cs(ir, wit)


if __name__ == "__main__":
    unittest.main()
