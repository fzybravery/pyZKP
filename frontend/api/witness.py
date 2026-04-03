from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

from common.ir.core import CircuitIR, Expr, Hint, LinExpr, VarRef
from frontend.api.hints import GLOBAL_HINTS, HintRegistry

# ZKP 的 witness，存储电路输入、输出、中间值
@dataclass(frozen=True)
class Witness:
    values: Tuple[int, ...]


def _inv_mod(x: int, p: int) -> int:
    x %= p
    if x == 0:
        return 0
    return pow(x, p - 2, p)

# 评估表达式，根据 witness 值计算表达式的值
def _eval_expr(modulus: int, values: List[Optional[int]], expr: Expr) -> Optional[int]:
    if isinstance(expr, int):
        return expr % modulus
    if isinstance(expr, VarRef):
        v = values[expr.id]
        if v is None:
            return None
        return v % modulus
    if isinstance(expr, LinExpr):
        acc = expr.const % modulus
        for vid, c in expr.terms:
            vv = values[vid]
            if vv is None:
                return None
            acc = (acc + (vv % modulus) * (c % modulus)) % modulus
        return acc
    raise TypeError(f"unsupported expr: {type(expr).__name__}")

# 收集赋值映射，将用户输入的赋值转换为 IR 中的变量 ID 映射
def _collect_assignment_map(assignment: Any) -> Dict[str, int]:
    if isinstance(assignment, Mapping):
        return {str(k): int(v) for k, v in assignment.items()}
    if hasattr(assignment, "__dict__"):
        out: Dict[str, int] = {}
        for k, v in assignment.__dict__.items():
            if isinstance(v, (int, bool)):
                out[str(k)] = int(v)
        return out
    raise TypeError("assignment must be dict-like or an object with __dict__")

# 构建 witness
def build_witness(ir: CircuitIR, assignment: Any, registry: HintRegistry | None = None) -> Witness:
    p = ir.field.modulus
    reg = registry or GLOBAL_HINTS
    asg = _collect_assignment_map(assignment)
    name_to_id = {v.name: v.id for v in ir.vars}

    values: List[Optional[int]] = [None] * len(ir.vars)

    for inp in ir.inputs:
        if inp.name not in asg:
            raise KeyError(f"missing input value: {inp.name}")
        if inp.name not in name_to_id:
            raise KeyError(f"missing input var in var table: {inp.name}")
        values[name_to_id[inp.name]] = int(asg[inp.name]) % p

    for h in ir.hints:
        _apply_hint(p, values, h, reg)

    _solve_constraints(p, values, ir.constraints)

    if any(v is None for v in values):
        missing = [i for i, v in enumerate(values) if v is None]
        raise ValueError(f"witness incomplete, missing var ids: {missing[:20]}")

    return Witness(values=tuple(cast(List[int], values)))

# 应用 hint 函数，根据 witness 值计算中间值
def _apply_hint(p: int, values: List[Optional[int]], h: Hint, reg: HintRegistry) -> None:
    op = h.op
    if op == "mul":
        if len(h.inputs) != 2 or len(h.outputs) != 1:
            raise ValueError("mul hint expects 2 inputs and 1 output")
        a = _eval_expr(p, values, h.inputs[0])
        b = _eval_expr(p, values, h.inputs[1])
        if a is None or b is None:
            raise ValueError("mul hint has unknown inputs")
        values[h.outputs[0]] = (a * b) % p
        return
    if op == "inv":
        if len(h.inputs) != 1 or len(h.outputs) != 1:
            raise ValueError("inv hint expects 1 input and 1 output")
        x = _eval_expr(p, values, h.inputs[0])
        if x is None:
            raise ValueError("inv hint has unknown inputs")
        values[h.outputs[0]] = _inv_mod(x, p)
        return
    if op == "is_zero":
        if len(h.inputs) != 1 or len(h.outputs) != 2:
            raise ValueError("is_zero hint expects 1 input and 2 outputs")
        a = _eval_expr(p, values, h.inputs[0])
        if a is None:
            raise ValueError("is_zero hint has unknown inputs")
        if a == 0:
            values[h.outputs[0]] = 1
            values[h.outputs[1]] = 0
        else:
            values[h.outputs[0]] = 0
            values[h.outputs[1]] = _inv_mod(a, p)
        return
    if op == "to_binary":
        if len(h.inputs) != 2:
            raise ValueError("to_binary hint expects 2 inputs")
        x = _eval_expr(p, values, h.inputs[0])
        n = _eval_expr(p, values, h.inputs[1])
        if x is None or n is None:
            raise ValueError("to_binary hint has unknown inputs")
        n_bits = int(n)
        if n_bits != len(h.outputs):
            raise ValueError("to_binary output length mismatch")
        xx = int(x)
        for i in range(n_bits):
            values[h.outputs[i]] = (xx >> i) & 1
        return
    if op == "assign":
        if len(h.inputs) != 1 or len(h.outputs) != 1:
            raise ValueError("assign hint expects 1 input and 1 output")
        x = _eval_expr(p, values, h.inputs[0])
        if x is None:
            raise ValueError("assign hint has unknown inputs")
        values[h.outputs[0]] = x % p
        return
    fn = reg.get(op)
    if fn is None:
        raise ValueError(f"unknown hint op: {op}")
    in_vals: List[int] = []
    for e in h.inputs:
        v = _eval_expr(p, values, e)
        if v is None:
            raise ValueError("user hint has unknown inputs")
        in_vals.append(int(v))
    out = fn(*in_vals)
    if isinstance(out, int):
        outs = [out]
    else:
        outs = list(out)
    if len(outs) != len(h.outputs):
        raise ValueError("user hint output arity mismatch")
    for vid, vv in zip(h.outputs, outs):
        values[vid] = int(vv) % p

# 检查 R1CS 约束是否满足
def check_r1cs(ir: CircuitIR, witness: Witness) -> None:
    p = ir.field.modulus
    values = list(witness.values)

    def eval_lin(le: LinExpr) -> int:
        acc = le.const % p
        for vid, c in le.terms:
            acc = (acc + (values[vid] % p) * (c % p)) % p
        return acc

    for i, c in enumerate(ir.constraints):
        a = eval_lin(c.a)
        b = eval_lin(c.b)
        cc = eval_lin(c.c)
        if (a * b - cc) % p != 0:
            raise AssertionError(f"constraint {i} unsatisfied")

# 迭代求解缺失的变量值
def _solve_constraints(p: int, values: List[Optional[int]], constraints: Tuple[Any, ...]) -> None:
    def is_const(le: LinExpr) -> bool:
        return len(le.terms) == 0

    def const_val(le: LinExpr) -> int:
        if not is_const(le):
            raise ValueError
        return le.const % p

    def eval_lin_partial(le: LinExpr) -> Optional[int]:
        acc = le.const % p
        for vid, c in le.terms:
            vv = values[vid]
            if vv is None:
                return None
            acc = (acc + (vv % p) * (c % p)) % p
        return acc

    def try_solve_linear_zero(le: LinExpr) -> bool:
        unknowns: List[Tuple[int, int]] = []
        acc = le.const % p
        for vid, c in le.terms:
            vv = values[vid]
            if vv is None:
                unknowns.append((vid, c % p))
            else:
                acc = (acc + (vv % p) * (c % p)) % p
        if len(unknowns) != 1:
            return False
        vid, coeff = unknowns[0]
        if coeff == 0:
            return False
        sol = (-acc) % p
        sol = (sol * _inv_mod(coeff, p)) % p
        values[vid] = sol
        return True

    max_rounds = len(constraints) + len(values) + 5
    for _ in range(max_rounds):
        progressed = False
        for c in constraints:
            a_le = c.a
            b_le = c.b
            c_le = c.c
            if is_const(b_le) and const_val(b_le) == 1 and is_const(c_le) and const_val(c_le) == 0:
                if try_solve_linear_zero(a_le):
                    progressed = True
                    continue
            av = eval_lin_partial(a_le)
            bv = eval_lin_partial(b_le)
            cv = eval_lin_partial(c_le)
            if av is not None and bv is not None and cv is None and len(c_le.terms) == 1 and c_le.const % p == 0:
                (vid, coeff) = c_le.terms[0]
                if values[vid] is None and coeff % p != 0:
                    values[vid] = ((av * bv) * _inv_mod(coeff, p)) % p
                    progressed = True
                    continue
            if av is None and bv is not None and cv is not None and len(a_le.terms) == 1 and a_le.const % p == 0:
                (vid, coeff) = a_le.terms[0]
                if values[vid] is None and coeff % p != 0 and bv % p != 0:
                    values[vid] = ((cv * _inv_mod(bv, p)) * _inv_mod(coeff, p)) % p
                    progressed = True
                    continue
        if not progressed:
            break
