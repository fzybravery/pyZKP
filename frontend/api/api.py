from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from pyZKP.common.ir.core import (
    Expr,
    Field,
    Hint,
    LinExpr,
    R1CSConstraint,
    Var as IRVar,
    VarRef,
    Visibility,
    as_linexpr,
    lin_add,
    lin_neg,
    lin_sub,
)
from pyZKP.frontend.api.hints import GLOBAL_HINTS, HintFn

"""
API 接口层，允许用户编写电路
"""


Var = Expr

# 电路，用户通过重写define方法来定义电路的逻辑
class Circuit:
    def define(self, api: "API") -> None:
        raise NotImplementedError

# 电路构建器
@dataclass
class _Builder:
    field: Field
    next_var_id: int
    vars: List[IRVar]
    constraints: List[R1CSConstraint]
    hints: List[Hint]

    def new_internal(self, name: str | None = None) -> VarRef:
        vid = self.next_var_id
        self.next_var_id += 1
        n = name or f"tmp{vid}"
        self.vars.append(IRVar(id=vid, name=n, visibility=Visibility.INTERNAL))
        return VarRef(vid)

    def add_hint(self, op: str, inputs: List[Expr], n_outputs: int, names: List[str] | None = None) -> List[VarRef]:
        outs: List[VarRef] = []
        out_ids: List[int] = []
        for i in range(n_outputs):
            name = None
            if names is not None and i < len(names):
                name = names[i]
            v = self.new_internal(name)
            outs.append(v)
            out_ids.append(v.id)
        self.hints.append(Hint(op=op, inputs=tuple(inputs), outputs=tuple(out_ids)))
        return outs

    def assign(self, out: VarRef, expr: Expr) -> None:
        self.hints.append(Hint(op="assign", inputs=(expr,), outputs=(out.id,)))
        self.assert_is_equal(out, expr)

    def add_r1cs(self, a: Expr, b: Expr, c: Expr) -> None:
        aa = as_linexpr(self.field, a)
        bb = as_linexpr(self.field, b)
        cc = as_linexpr(self.field, c)
        self.constraints.append(R1CSConstraint(a=aa, b=bb, c=cc))

    def assert_is_equal(self, x: Expr, y: Expr) -> None:
        self.add_r1cs(lin_sub(self.field, as_linexpr(self.field, x), as_linexpr(self.field, y)), 1, 0)

    def assert_is_boolean(self, x: Expr) -> None:
        lx = as_linexpr(self.field, x)
        self.add_r1cs(lx, lin_sub(self.field, lx, as_linexpr(self.field, 1)), 0)


class API:
    def __init__(self, builder: _Builder):
        self._b = builder

    def _as_expr(self, v: Var) -> Expr:
        if isinstance(v, (int, VarRef, LinExpr)):
            return v
        raise TypeError(f"unsupported variable type: {type(v).__name__}")

    def Constant(self, v: int) -> Expr:
        return self._b.field.normalize(v)

    # 允许用户自定义 hint 函数
    def Hint(self, fn_or_name: str | HintFn, inputs: Sequence[Var], n_outputs: int = 1, names: List[str] | None = None) -> Var | List[Var]:
        if isinstance(fn_or_name, str):
            op = fn_or_name
        else:
            fn = fn_or_name
            qual = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or "hint"
            mod = getattr(fn, "__module__", "user")
            op = f"user:{mod}.{qual}"
            GLOBAL_HINTS.register(op, fn)
        outs = self._b.add_hint(op, [self._as_expr(x) for x in inputs], n_outputs=n_outputs, names=names)
        return outs[0] if n_outputs == 1 else outs

    def Add(self, i1: Var, i2: Var, *rest: Var) -> Expr:
        acc = lin_add(self._b.field, as_linexpr(self._b.field, self._as_expr(i1)), as_linexpr(self._b.field, self._as_expr(i2)))
        for x in rest:
            acc = lin_add(self._b.field, acc, as_linexpr(self._b.field, self._as_expr(x)))
        return acc

    def Sub(self, i1: Var, i2: Var, *rest: Var) -> Expr:
        acc = lin_sub(self._b.field, as_linexpr(self._b.field, self._as_expr(i1)), as_linexpr(self._b.field, self._as_expr(i2)))
        for x in rest:
            acc = lin_sub(self._b.field, acc, as_linexpr(self._b.field, self._as_expr(x)))
        return acc

    def Neg(self, i1: Var) -> Expr:
        return lin_neg(self._b.field, as_linexpr(self._b.field, self._as_expr(i1)))

    def Mul(self, i1: Var, i2: Var, *rest: Var) -> Expr:
        def mul2(a: Expr, b: Expr) -> VarRef:
            out = self._b.add_hint("mul", [a, b], n_outputs=1, names=["mul"])[0]
            self._b.add_r1cs(a, b, out)
            return out

        acc: Expr = mul2(self._as_expr(i1), self._as_expr(i2))
        for x in rest:
            acc = mul2(acc, self._as_expr(x))
        return acc

    def MulAcc(self, a: Var, b: Var, c: Var) -> Expr:
        return self.Add(a, self.Mul(b, c))

    def Inverse(self, i1: Var) -> VarRef:
        x = self._as_expr(i1)
        inv = self._b.add_hint("inv", [x], n_outputs=1, names=["inv"])[0]
        self._b.add_r1cs(x, inv, 1)
        return inv

    def Div(self, i1: Var, i2: Var) -> VarRef:
        inv = self.Inverse(i2)
        return self.Mul(i1, inv)  # type: ignore[return-value]

    def ToBinary(self, i1: Var, n_bits: int) -> List[VarRef]:
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        x = self._as_expr(i1)
        bits = self._b.add_hint("to_binary", [x, int(n_bits)], n_outputs=n_bits, names=[f"bit{i}" for i in range(n_bits)])
        for b in bits:
            self._b.assert_is_boolean(b)
        coeffs = {b.id: (1 << i) for i, b in enumerate(bits)}
        recomposed = LinExpr.from_terms(0, coeffs)
        self._b.assert_is_equal(x, recomposed)
        return bits

    def FromBinary(self, bits: Sequence[Var]) -> LinExpr:
        coeffs = {self._as_expr(b).id: (1 << i) for i, b in enumerate(bits) if isinstance(self._as_expr(b), VarRef)}
        if len(coeffs) != len(bits):
            raise TypeError("FromBinary expects bits to be VarRef")
        return LinExpr.from_terms(0, coeffs)

    def AssertIsEqual(self, i1: Var, i2: Var) -> None:
        self._b.assert_is_equal(self._as_expr(i1), self._as_expr(i2))

    def AssertIsDifferent(self, i1: Var, i2: Var) -> None:
        diff = self.Sub(i1, i2)
        inv = self._b.add_hint("inv", [diff], n_outputs=1, names=["inv_diff"])[0]
        self._b.add_r1cs(diff, inv, 1)

    def AssertIsBoolean(self, v: Var) -> None:
        self._b.assert_is_boolean(self._as_expr(v))

    def IsZero(self, i1: Var) -> VarRef:
        a = self._as_expr(i1)
        z, inv = self._b.add_hint("is_zero", [a], n_outputs=2, names=["is_zero", "inv_is_zero"])
        self._b.assert_is_boolean(z)
        self._b.add_r1cs(a, z, 0)
        self._b.add_r1cs(a, inv, self.Sub(1, z))
        return z

    def Select(self, b: Var, i1: Var, i2: Var) -> VarRef:
        bb = self._as_expr(b)
        self._b.assert_is_boolean(bb)
        x = as_linexpr(self._b.field, self._as_expr(i1))
        y = as_linexpr(self._b.field, self._as_expr(i2))
        diff = lin_sub(self._b.field, x, y)
        t = self._b.add_hint("mul", [bb, diff], n_outputs=1, names=["sel_mul"])[0]
        self._b.add_r1cs(bb, diff, t)
        out = self._b.new_internal("sel_out")
        self._b.assign(out, self.Add(y, t))
        return out

    def And(self, a: Var, b: Var) -> VarRef:
        aa = self._as_expr(a)
        bb = self._as_expr(b)
        self._b.assert_is_boolean(aa)
        self._b.assert_is_boolean(bb)
        out = self._b.add_hint("mul", [aa, bb], n_outputs=1, names=["and"])[0]
        self._b.add_r1cs(aa, bb, out)
        self._b.assert_is_boolean(out)
        return out

    def Or(self, a: Var, b: Var) -> VarRef:
        aa = self._as_expr(a)
        bb = self._as_expr(b)
        self._b.assert_is_boolean(aa)
        self._b.assert_is_boolean(bb)
        prod = self.Mul(aa, bb)
        out = self._b.new_internal("or")
        self._b.assign(out, self.Sub(self.Add(aa, bb), prod))
        self._b.assert_is_boolean(out)
        return out

    def Xor(self, a: Var, b: Var) -> VarRef:
        aa = self._as_expr(a)
        bb = self._as_expr(b)
        self._b.assert_is_boolean(aa)
        self._b.assert_is_boolean(bb)
        prod = self.Mul(aa, bb)
        out = self._b.new_internal("xor")
        self._b.assign(out, self.Sub(self.Add(aa, bb), self.Mul(2, prod)))
        self._b.assert_is_boolean(out)
        return out
