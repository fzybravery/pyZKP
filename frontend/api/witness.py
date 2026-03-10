from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from pyZKP.common.ir.core import CircuitIR, Expr, Hint, LinExpr, R1CSConstraint, VarRef, as_linexpr


@dataclass(frozen=True)
class Witness:
    values: Tuple[int, ...]


def _inv_mod(x: int, p: int) -> int:
    x %= p
    if x == 0:
        return 0
    return pow(x, p - 2, p)


def _eval_expr(modulus: int, values: List[int], expr: Expr) -> int:
    if isinstance(expr, int):
        return expr % modulus
    if isinstance(expr, VarRef):
        return values[expr.id] % modulus
    if isinstance(expr, LinExpr):
        acc = expr.const % modulus
        for vid, c in expr.terms:
            acc = (acc + (values[vid] % modulus) * (c % modulus)) % modulus
        return acc
    raise TypeError(f"unsupported expr: {type(expr).__name__}")


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


def build_witness(ir: CircuitIR, assignment: Any) -> Witness:
    p = ir.field.modulus
    asg = _collect_assignment_map(assignment)
    name_to_id = {v.name: v.id for v in ir.vars}

    values: List[int] = [0] * len(ir.vars)

    for inp in ir.inputs:
        if inp.name not in asg:
            raise KeyError(f"missing input value: {inp.name}")
        if inp.name not in name_to_id:
            raise KeyError(f"missing input var in var table: {inp.name}")
        values[name_to_id[inp.name]] = int(asg[inp.name]) % p

    for h in ir.hints:
        _apply_hint(p, values, h)

    return Witness(values=tuple(values))


def _apply_hint(p: int, values: List[int], h: Hint) -> None:
    op = h.op
    if op == "mul":
        if len(h.inputs) != 2 or len(h.outputs) != 1:
            raise ValueError("mul hint expects 2 inputs and 1 output")
        a = _eval_expr(p, values, h.inputs[0])
        b = _eval_expr(p, values, h.inputs[1])
        values[h.outputs[0]] = (a * b) % p
        return
    if op == "inv":
        if len(h.inputs) != 1 or len(h.outputs) != 1:
            raise ValueError("inv hint expects 1 input and 1 output")
        x = _eval_expr(p, values, h.inputs[0])
        values[h.outputs[0]] = _inv_mod(x, p)
        return
    if op == "is_zero":
        if len(h.inputs) != 1 or len(h.outputs) != 2:
            raise ValueError("is_zero hint expects 1 input and 2 outputs")
        a = _eval_expr(p, values, h.inputs[0])
        if a == 0:
            values[h.outputs[0]] = 1
            values[h.outputs[1]] = 0
        else:
            values[h.outputs[0]] = 0
            values[h.outputs[1]] = _inv_mod(a, p)
        return
    if op == "assign":
        if len(h.inputs) != 1 or len(h.outputs) != 1:
            raise ValueError("assign hint expects 1 input and 1 output")
        x = _eval_expr(p, values, h.inputs[0])
        values[h.outputs[0]] = x % p
        return
    raise ValueError(f"unknown hint op: {op}")


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
