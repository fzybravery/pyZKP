from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.ir.core import CircuitIR, LinExpr


@dataclass(frozen=True)
class R1CSInstance:
    n_constraints: int
    n_vars: int
    one_id: int
    a_rows: List[Dict[int, int]]
    b_rows: List[Dict[int, int]]
    c_rows: List[Dict[int, int]]
    public_ids: List[int]
    public_names: List[str]
    aux_ids: List[int]


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compile_r1cs(ir: CircuitIR) -> R1CSInstance:
    m = len(ir.vars)
    one_id = m
    n0 = len(ir.constraints)
    n = _next_power_of_two(n0)
    pub_ids = sorted([v.id for v in ir.vars if v.visibility.value == "public"])
    pub_names = [v.name for v in sorted([v for v in ir.vars if v.visibility.value == "public"], key=lambda x: x.id)]
    public_ids = [one_id] + pub_ids
    public_names = ["ONE"] + pub_names
    aux_ids = [i for i in range(m) if i not in pub_ids]

    a_rows: List[Dict[int, int]] = []
    b_rows: List[Dict[int, int]] = []
    c_rows: List[Dict[int, int]] = []

    for c in ir.constraints:
        a_rows.append(_linexpr_to_row(c.a, one_id))
        b_rows.append(_linexpr_to_row(c.b, one_id))
        c_rows.append(_linexpr_to_row(c.c, one_id))

    while len(a_rows) < n:
        a_rows.append({})
        b_rows.append({})
        c_rows.append({})

    return R1CSInstance(
        n_constraints=n,
        n_vars=m + 1,
        one_id=one_id,
        a_rows=a_rows,
        b_rows=b_rows,
        c_rows=c_rows,
        public_ids=public_ids,
        public_names=public_names,
        aux_ids=aux_ids,
    )


def eval_row(row: Dict[int, int], w: Sequence[int]) -> int:
    acc = 0
    for vid, coeff in row.items():
        acc = (acc + (w[vid] % FR_MODULUS) * (coeff % FR_MODULUS)) % FR_MODULUS
    return acc % FR_MODULUS


def eval_r1cs_vectors(r1cs: R1CSInstance, w: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    a: List[int] = []
    b: List[int] = []
    c: List[int] = []
    for ar, br, cr in zip(r1cs.a_rows, r1cs.b_rows, r1cs.c_rows):
        a.append(eval_row(ar, w))
        b.append(eval_row(br, w))
        c.append(eval_row(cr, w))
    return a, b, c


def _linexpr_to_row(le: LinExpr, one_id: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for vid, coeff in le.terms:
        out[vid] = (out.get(vid, 0) + coeff) % FR_MODULUS
    if le.const % FR_MODULUS != 0:
        out[one_id] = (out.get(one_id, 0) + le.const) % FR_MODULUS
    return {k: v for k, v in out.items() if v % FR_MODULUS != 0}
