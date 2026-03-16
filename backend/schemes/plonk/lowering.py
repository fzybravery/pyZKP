from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from pyZKP.backend.schemes.plonk.domain import find_coset_factors, next_power_of_two, roots_of_unity
from pyZKP.backend.schemes.plonk.types import Circuit, Domain, Gate
from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.ir.core import CircuitIR, LinExpr


def lower_to_circuit(ir: CircuitIR) -> Circuit:
    m0 = len(ir.vars)
    one_id = m0
    zero_id = one_id + 1
    next_id = zero_id + 1

    public_vars = [v for v in ir.vars if v.visibility.value == "public"]
    public_vars_sorted = sorted(public_vars, key=lambda v: v.id)
    public_var_ids = [v.id for v in public_vars_sorted]
    public_names = [v.name for v in public_vars_sorted]

    gates: List[Gate] = []

    gates.append(Gate(l=one_id, r=one_id, o=zero_id, ql=0, qr=0, qm=0, qo=1, qc=0, kind="const_zero"))

    for pid in public_var_ids:
        gates.append(Gate(l=pid, r=one_id, o=one_id, ql=1, qr=0, qm=0, qo=0, qc=0, kind="public"))

    def new_var(name: str) -> int:
        nonlocal next_id
        vid = next_id
        next_id += 1
        return vid

    def make_const(c: int) -> int:
        if c % FR_MODULUS == 0:
            return zero_id
        out = new_var("const")
        gates.append(Gate(l=one_id, r=one_id, o=out, ql=0, qr=0, qm=0, qo=1, qc=(-c) % FR_MODULUS, kind="const"))
        return out

    def scale_var(var_id: int, coeff: int) -> int:
        cc = coeff % FR_MODULUS
        if cc == 0:
            return zero_id
        if cc == 1:
            return var_id
        out = new_var("scale")
        gates.append(Gate(l=var_id, r=one_id, o=out, ql=cc, qr=0, qm=0, qo=(-1) % FR_MODULUS, qc=0, kind="scale"))
        return out

    def add_vars(a: int, b: int) -> int:
        if a == zero_id:
            return b
        if b == zero_id:
            return a
        out = new_var("add")
        gates.append(Gate(l=a, r=b, o=out, ql=1, qr=1, qm=0, qo=(-1) % FR_MODULUS, qc=0, kind="add"))
        return out

    def materialize(le: LinExpr) -> int:
        acc = make_const(le.const)
        for vid, coeff in le.terms:
            term = scale_var(vid, coeff)
            acc = add_vars(acc, term)
        return acc

    for con in ir.constraints:
        a = materialize(con.a)
        b = materialize(con.b)
        c = materialize(con.c)
        gates.append(Gate(l=a, r=b, o=c, ql=0, qr=0, qm=1, qo=(-1) % FR_MODULUS, qc=0, kind="r1cs_mul"))

    n_rows = len(gates)
    n = next_power_of_two(n_rows)
    omega, k1, k2 = find_coset_factors(n)
    roots = roots_of_unity(n, omega)
    domain = Domain(n=n, omega=omega, roots=roots)

    while len(gates) < n:
        a = new_var("pad_l")
        b = new_var("pad_r")
        o = new_var("pad_o")
        gates.append(Gate(l=a, r=b, o=o, ql=0, qr=0, qm=0, qo=0, qc=0, kind="pad"))

    sigma1, sigma2, sigma3 = _build_sigmas(gates, roots, k1, k2)

    ql = tuple(int(g.ql) % FR_MODULUS for g in gates)
    qr = tuple(int(g.qr) % FR_MODULUS for g in gates)
    qm = tuple(int(g.qm) % FR_MODULUS for g in gates)
    qo = tuple(int(g.qo) % FR_MODULUS for g in gates)
    qc = tuple(int(g.qc) % FR_MODULUS for g in gates)

    return Circuit(
        domain=domain,
        one_id=one_id,
        public_var_ids=public_var_ids,
        public_names=public_names,
        gates=tuple(gates),
        k1=k1,
        k2=k2,
        sigma1_eval=tuple(sigma1),
        sigma2_eval=tuple(sigma2),
        sigma3_eval=tuple(sigma3),
        ql_eval=ql,
        qr_eval=qr,
        qm_eval=qm,
        qo_eval=qo,
        qc_eval=qc,
    )


def _build_sigmas(gates: List[Gate], roots: Tuple[int, ...], k1: int, k2: int) -> Tuple[List[int], List[int], List[int]]:
    n = len(roots)
    pos_vars: List[int] = []
    id_vals: List[int] = []
    for i in range(n):
        g = gates[i]
        pos_vars.extend([g.l, g.r, g.o])
        id_vals.extend([roots[i], (k1 * roots[i]) % FR_MODULUS, (k2 * roots[i]) % FR_MODULUS])

    groups: Dict[int, List[int]] = {}
    for p, vid in enumerate(pos_vars):
        groups.setdefault(vid, []).append(p)

    perm: List[int] = list(range(3 * n))
    for _, positions in groups.items():
        if len(positions) <= 1:
            continue
        for i, p in enumerate(positions):
            perm[p] = positions[(i + 1) % len(positions)]

    sigma_vals = [id_vals[perm[p]] for p in range(3 * n)]
    sigma1 = [sigma_vals[3 * i + 0] for i in range(n)]
    sigma2 = [sigma_vals[3 * i + 1] for i in range(n)]
    sigma3 = [sigma_vals[3 * i + 2] for i in range(n)]
    return sigma1, sigma2, sigma3
