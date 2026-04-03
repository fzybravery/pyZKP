from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Union


class Visibility(str, Enum):
    PUBLIC = "public"
    SECRET = "secret"
    INTERNAL = "internal"
    CONST = "const"


@dataclass(frozen=True)
class Field:
    modulus: int

    def normalize(self, x: int) -> int:
        return x % self.modulus


@dataclass(frozen=True)
class Var:
    id: int
    name: str
    visibility: Visibility


@dataclass(frozen=True)
class Input:
    name: str
    visibility: Visibility


@dataclass(frozen=True)
class VarRef:
    id: int


@dataclass(frozen=True)
class LinExpr:
    const: int = 0
    terms: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)

    @staticmethod
    def from_terms(const: int, coeffs: Dict[int, int]) -> LinExpr:
        items = tuple(sorted(((vid, c) for vid, c in coeffs.items() if c != 0), key=lambda t: t[0]))
        return LinExpr(const=const, terms=items)

    def to_dict(self) -> Dict[int, int]:
        return {vid: c for vid, c in self.terms}

@dataclass(frozen=True)
class R1CSConstraint:
    a: LinExpr
    b: LinExpr
    c: LinExpr


@dataclass(frozen=True)
class Hint:
    op: str
    inputs: Tuple["Expr", ...]
    outputs: Tuple[int, ...]


@dataclass(frozen=True)
class CircuitIR:
    field: Field
    inputs: Tuple[Input, ...]
    vars: Tuple[Var, ...]
    constraints: Tuple[R1CSConstraint, ...]
    hints: Tuple[Hint, ...] = field(default_factory=tuple)


Expr = Union[int, VarRef, LinExpr]


def is_const(x: Expr) -> bool:
    return isinstance(x, int)


def is_var(x: Expr) -> bool:
    return isinstance(x, VarRef)


def as_linexpr(field: Field, x: Expr) -> LinExpr:
    if isinstance(x, LinExpr):
        return x
    if isinstance(x, VarRef):
        return LinExpr.from_terms(0, {x.id: 1})
    if isinstance(x, int):
        return LinExpr(const=field.normalize(x))
    raise TypeError(f"not linear: {type(x).__name__}")


def lin_add(field: Field, a: LinExpr, b: LinExpr) -> LinExpr:
    out: Dict[int, int] = {}
    for vid, c in a.terms:
        out[vid] = (out.get(vid, 0) + c) % field.modulus
    for vid, c in b.terms:
        out[vid] = (out.get(vid, 0) + c) % field.modulus
    const = (a.const + b.const) % field.modulus
    return LinExpr.from_terms(const, out)


def lin_neg(field: Field, a: LinExpr) -> LinExpr:
    out = {vid: (-c) % field.modulus for vid, c in a.terms}
    const = (-a.const) % field.modulus
    return LinExpr.from_terms(const, out)


def lin_sub(field: Field, a: LinExpr, b: LinExpr) -> LinExpr:
    return lin_add(field, a, lin_neg(field, b))


def lin_scale(field: Field, a: LinExpr, k: int) -> LinExpr:
    kk = k % field.modulus
    out = {vid: (c * kk) % field.modulus for vid, c in a.terms}
    const = (a.const * kk) % field.modulus
    return LinExpr.from_terms(const, out)


def collect_vars(expr: Expr) -> List[int]:
    out: List[int] = []
    stack: List[Expr] = [expr]
    while stack:
        cur = stack.pop()
        if isinstance(cur, VarRef):
            out.append(cur.id)
        elif isinstance(cur, LinExpr):
            out.extend([vid for vid, _ in cur.terms])
        elif isinstance(cur, int):
            continue
        else:
            raise TypeError(f"unknown expr: {type(cur).__name__}")
    return out


def ensure_expr_is_ir_compatible(expr: Expr) -> None:
    stack: List[Expr] = [expr]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (int, VarRef, LinExpr)):
            continue
        raise TypeError(f"unknown expr: {type(cur).__name__}")
