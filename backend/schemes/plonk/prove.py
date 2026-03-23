from __future__ import annotations

import secrets
from typing import Dict, List, Sequence, Tuple

from pyZKP.backend.schemes.plonk.transcript import Transcript
from pyZKP.backend.schemes.plonk.types import Proof, ProvingKey
from pyZKP.common.crypto.ecc.bn254 import G1, G1_ZERO, g1_add, g1_mul
from pyZKP.common.crypto.field import fr_batch_inv
from pyZKP.common.crypto.field.fr import FR_MODULUS, fr_inv
from pyZKP.common.crypto.kzg.cpu_ref import commit, open_proof
from pyZKP.common.crypto.poly import (
    coeffs_from_evals_on_roots,
    coeffs_from_evals_on_coset,
    evals_from_coeffs_on_coset,
    omega_for_size,
    poly_eval,
)
from pyZKP.frontend.api.witness import Witness


def prove(pk: ProvingKey, witness: Witness, public_values: Sequence[int]) -> Proof:
    c = pk.circuit
    n = c.domain.n
    omega = c.domain.omega

    if len(public_values) != 1 + len(c.public_var_ids):
        raise ValueError("public_values must be [ONE] + public inputs in schema order")
    if int(public_values[0]) % FR_MODULUS != 1:
        raise ValueError("public_values[0] must be ONE==1")

    values = _build_extended_values(c, witness)
    a_eval = [values[g.l] for g in c.gates]
    b_eval = [values[g.r] for g in c.gates]
    c_eval = [values[g.o] for g in c.gates]

    a_coeff = tuple(coeffs_from_evals_on_roots(a_eval, omega=omega))
    b_coeff = tuple(coeffs_from_evals_on_roots(b_eval, omega=omega))
    c_coeff = tuple(coeffs_from_evals_on_roots(c_eval, omega=omega))

    cm_a = commit(pk.srs, a_coeff)
    cm_b = commit(pk.srs, b_coeff)
    cm_c = commit(pk.srs, c_coeff)

    pi_eval = [0] * n
    for i, pv in enumerate(public_values[1:]):
        row = 1 + i
        pi_eval[row] = (-int(pv)) % FR_MODULUS
    pi_coeff = tuple(coeffs_from_evals_on_roots(pi_eval, omega=omega))

    tr = Transcript()
    tr.absorb_g1(pk.vk.cm_sigma1)
    tr.absorb_g1(pk.vk.cm_sigma2)
    tr.absorb_g1(pk.vk.cm_sigma3)
    tr.absorb_g1(pk.vk.cm_ql)
    tr.absorb_g1(pk.vk.cm_qr)
    tr.absorb_g1(pk.vk.cm_qm)
    tr.absorb_g1(pk.vk.cm_qo)
    tr.absorb_g1(pk.vk.cm_qc)
    for x in public_values:
        tr.absorb_int(int(x))
    tr.absorb_g1(cm_a)
    tr.absorb_g1(cm_b)
    tr.absorb_g1(cm_c)

    beta = tr.challenge(b"beta")
    gamma = tr.challenge(b"gamma")

    z_eval = _build_permutation_z(c, a_eval, b_eval, c_eval, beta, gamma)
    z_coeff = tuple(coeffs_from_evals_on_roots(z_eval, omega=omega))
    cm_z = commit(pk.srs, z_coeff)

    tr.absorb_g1(cm_z)
    alpha = tr.challenge(b"alpha")

    t1_coeff, t2_coeff, t3_coeff = _compute_quotient_t_parts(
        pk=pk,
        a_coeff=a_coeff,
        b_coeff=b_coeff,
        c_coeff=c_coeff,
        z_coeff=z_coeff,
        z_eval=z_eval,
        pi_coeff=pi_coeff,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    cm_t1 = commit(pk.srs, t1_coeff)
    cm_t2 = commit(pk.srs, t2_coeff)
    cm_t3 = commit(pk.srs, t3_coeff)

    tr.absorb_g1(cm_t1)
    tr.absorb_g1(cm_t2)
    tr.absorb_g1(cm_t3)
    zeta = tr.challenge(b"zeta")
    zeta_omega = (zeta * omega) % FR_MODULUS

    evals = _collect_evals(pk, a_coeff, b_coeff, c_coeff, z_coeff, t1_coeff, t2_coeff, t3_coeff, pi_coeff, zeta)
    z_zw = poly_eval(z_coeff, zeta_omega)

    tr.absorb_int(evals["a"])
    tr.absorb_int(evals["b"])
    tr.absorb_int(evals["c"])
    tr.absorb_int(evals["z"])
    tr.absorb_int(z_zw)
    tr.absorb_int(evals["t1"])
    tr.absorb_int(evals["t2"])
    tr.absorb_int(evals["t3"])
    tr.absorb_int(evals["s1"])
    tr.absorb_int(evals["s2"])
    tr.absorb_int(evals["s3"])
    tr.absorb_int(evals["ql"])
    tr.absorb_int(evals["qr"])
    tr.absorb_int(evals["qm"])
    tr.absorb_int(evals["qo"])
    tr.absorb_int(evals["qc"])
    v = tr.challenge(b"v")

    polys = [
        ("a", a_coeff, cm_a, evals["a"]),
        ("b", b_coeff, cm_b, evals["b"]),
        ("c", c_coeff, cm_c, evals["c"]),
        ("z", z_coeff, cm_z, evals["z"]),
        ("t1", t1_coeff, cm_t1, evals["t1"]),
        ("t2", t2_coeff, cm_t2, evals["t2"]),
        ("t3", t3_coeff, cm_t3, evals["t3"]),
        ("s1", pk.coeff_sigma1, pk.vk.cm_sigma1, evals["s1"]),
        ("s2", pk.coeff_sigma2, pk.vk.cm_sigma2, evals["s2"]),
        ("s3", pk.coeff_sigma3, pk.vk.cm_sigma3, evals["s3"]),
        ("ql", pk.coeff_ql, pk.vk.cm_ql, evals["ql"]),
        ("qr", pk.coeff_qr, pk.vk.cm_qr, evals["qr"]),
        ("qm", pk.coeff_qm, pk.vk.cm_qm, evals["qm"]),
        ("qo", pk.coeff_qo, pk.vk.cm_qo, evals["qo"]),
        ("qc", pk.coeff_qc, pk.vk.cm_qc, evals["qc"]),
    ]
    combined_coeff, combined_cm, combined_y = _fold(polys, v)
    y_check, pi_zeta = open_proof(pk.srs, combined_coeff, zeta)
    if y_check % FR_MODULUS != combined_y % FR_MODULUS:
        raise ValueError("batch opening mismatch")

    _, pi_zeta_omega = open_proof(pk.srs, z_coeff, zeta_omega)

    return Proof(
        cm_a=cm_a,
        cm_b=cm_b,
        cm_c=cm_c,
        cm_z=cm_z,
        cm_t1=cm_t1,
        cm_t2=cm_t2,
        cm_t3=cm_t3,
        evals_zeta=evals,
        eval_zeta_omega=z_zw % FR_MODULUS,
        pi_zeta=pi_zeta,
        pi_zeta_omega=pi_zeta_omega,
    )


def _build_extended_values(c, witness: Witness) -> List[int]:
    max_id = c.one_id
    for g in c.gates:
        max_id = max(max_id, g.l, g.r, g.o)
    values: List[int | None] = [None] * (max_id + 1)

    base = list(witness.values)
    for i, v in enumerate(base):
        values[i] = int(v) % FR_MODULUS
    values[c.one_id] = 1

    for g in c.gates:
        l = values[g.l] if g.l <= max_id else None
        r = values[g.r] if g.r <= max_id else None
        if g.kind == "const_zero":
            values[g.o] = 0
        elif g.kind == "const":
            values[g.o] = (-g.qc) % FR_MODULUS
        elif g.kind == "scale":
            if l is None:
                raise ValueError("scale input unknown")
            values[g.o] = (g.ql * l) % FR_MODULUS
        elif g.kind == "add":
            if l is None or r is None:
                raise ValueError("add input unknown")
            values[g.o] = (l + r) % FR_MODULUS
        elif g.kind == "r1cs_mul":
            if values[g.o] is None and l is not None and r is not None:
                values[g.o] = (l * r) % FR_MODULUS

    missing = [i for i, v in enumerate(values) if v is None]
    for i in missing:
        values[i] = 0
    return [int(v) for v in values]  # type: ignore[arg-type]


def _build_permutation_z(c, a_eval, b_eval, c_eval, beta: int, gamma: int) -> List[int]:
    n = c.domain.n
    roots = c.domain.roots
    z = [0] * n
    z[0] = 1
    acc = 1
    for i in range(n - 1):
        x = roots[i]
        id1 = x
        id2 = (c.k1 * x) % FR_MODULUS
        id3 = (c.k2 * x) % FR_MODULUS
        num = (a_eval[i] + beta * id1 + gamma) % FR_MODULUS
        num = (num * (b_eval[i] + beta * id2 + gamma)) % FR_MODULUS
        num = (num * (c_eval[i] + beta * id3 + gamma)) % FR_MODULUS

        den = (a_eval[i] + beta * c.sigma1_eval[i] + gamma) % FR_MODULUS
        den = (den * (b_eval[i] + beta * c.sigma2_eval[i] + gamma)) % FR_MODULUS
        den = (den * (c_eval[i] + beta * c.sigma3_eval[i] + gamma)) % FR_MODULUS
        acc = (acc * num) % FR_MODULUS
        acc = (acc * fr_inv(den)) % FR_MODULUS
        z[i + 1] = acc
    return z


def _compute_quotient_t_parts(
    *,
    pk: ProvingKey,
    a_coeff: Tuple[int, ...],
    b_coeff: Tuple[int, ...],
    c_coeff: Tuple[int, ...],
    z_coeff: Tuple[int, ...],
    z_eval: Sequence[int],
    pi_coeff: Tuple[int, ...],
    alpha: int,
    beta: int,
    gamma: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    c = pk.circuit
    n = c.domain.n
    omega = c.domain.omega

    m = 4 * n
    omega_m = omega_for_size(m)
    u = pow(int(omega_m) % FR_MODULUS, n, FR_MODULUS)

    bad = {1 % FR_MODULUS, u % FR_MODULUS, (u * u) % FR_MODULUS, (u * u * u) % FR_MODULUS}
    while True:
        shift = secrets.randbelow(FR_MODULUS - 1) + 1
        shift_n = pow(shift, n, FR_MODULUS)
        if shift_n not in bad:
            break

    a_ext = evals_from_coeffs_on_coset(a_coeff, n=m, omega=omega_m, shift=shift)
    b_ext = evals_from_coeffs_on_coset(b_coeff, n=m, omega=omega_m, shift=shift)
    c_ext = evals_from_coeffs_on_coset(c_coeff, n=m, omega=omega_m, shift=shift)
    z_ext = evals_from_coeffs_on_coset(z_coeff, n=m, omega=omega_m, shift=shift)
    pi_ext = evals_from_coeffs_on_coset(pi_coeff, n=m, omega=omega_m, shift=shift)

    ql_ext = evals_from_coeffs_on_coset(pk.coeff_ql, n=m, omega=omega_m, shift=shift)
    qr_ext = evals_from_coeffs_on_coset(pk.coeff_qr, n=m, omega=omega_m, shift=shift)
    qm_ext = evals_from_coeffs_on_coset(pk.coeff_qm, n=m, omega=omega_m, shift=shift)
    qo_ext = evals_from_coeffs_on_coset(pk.coeff_qo, n=m, omega=omega_m, shift=shift)
    qc_ext = evals_from_coeffs_on_coset(pk.coeff_qc, n=m, omega=omega_m, shift=shift)

    s1_ext = evals_from_coeffs_on_coset(pk.coeff_sigma1, n=m, omega=omega_m, shift=shift)
    s2_ext = evals_from_coeffs_on_coset(pk.coeff_sigma2, n=m, omega=omega_m, shift=shift)
    s3_ext = evals_from_coeffs_on_coset(pk.coeff_sigma3, n=m, omega=omega_m, shift=shift)

    shift_zw = (int(shift) * int(omega)) % FR_MODULUS
    zshift_ext = evals_from_coeffs_on_coset(z_coeff, n=m, omega=omega_m, shift=shift_zw)

    l1_eval = [1] + [0] * (n - 1)
    l1_coeff = coeffs_from_evals_on_roots(l1_eval, omega=omega)
    l1_ext = evals_from_coeffs_on_coset(l1_coeff, n=m, omega=omega_m, shift=shift)

    alpha = int(alpha) % FR_MODULUS
    beta = int(beta) % FR_MODULUS
    gamma = int(gamma) % FR_MODULUS
    alpha2 = (alpha * alpha) % FR_MODULUS
    k1 = int(c.k1) % FR_MODULUS
    k2 = int(c.k2) % FR_MODULUS

    roots_m = 1
    x = shift % FR_MODULUS
    shift_n = pow(int(shift) % FR_MODULUS, n, FR_MODULUS)

    zh_list: List[int] = []
    num_list: List[int] = []
    for i in range(m):
        a = a_ext[i]
        b = b_ext[i]
        cc = c_ext[i]
        z = z_ext[i]

        gate = (ql_ext[i] * a + qr_ext[i] * b) % FR_MODULUS
        gate = (gate + qm_ext[i] * a % FR_MODULUS * b) % FR_MODULUS
        gate = (gate + qo_ext[i] * cc + qc_ext[i] + pi_ext[i]) % FR_MODULUS

        t1 = (a + beta * x + gamma) % FR_MODULUS
        t2 = (b + beta * k1 % FR_MODULUS * x + gamma) % FR_MODULUS
        t3 = (cc + beta * k2 % FR_MODULUS * x + gamma) % FR_MODULUS
        left = (t1 * t2) % FR_MODULUS
        left = (left * t3) % FR_MODULUS
        left = (left * z) % FR_MODULUS

        u1v = (a + beta * s1_ext[i] + gamma) % FR_MODULUS
        u2v = (b + beta * s2_ext[i] + gamma) % FR_MODULUS
        u3v = (cc + beta * s3_ext[i] + gamma) % FR_MODULUS
        right = (u1v * u2v) % FR_MODULUS
        right = (right * u3v) % FR_MODULUS
        right = (right * zshift_ext[i]) % FR_MODULUS

        perm = (left - right) % FR_MODULUS
        boundary = ((z - 1) % FR_MODULUS) * (l1_ext[i] % FR_MODULUS) % FR_MODULUS

        num = (gate + alpha * perm + alpha2 * boundary) % FR_MODULUS
        num_list.append(num)

        x_n = (shift_n * pow(u, i, FR_MODULUS)) % FR_MODULUS
        zh = (x_n - 1) % FR_MODULUS
        zh_list.append(zh)

        roots_m = (roots_m * omega_m) % FR_MODULUS
        x = (x * omega_m) % FR_MODULUS

    inv_zh = fr_batch_inv(zh_list)
    t_ext = [(num_list[i] * inv_zh[i]) % FR_MODULUS for i in range(m)]

    t_coeff_full = coeffs_from_evals_on_coset(t_ext, omega=omega_m, shift=shift)
    for v in t_coeff_full[3 * n :]:
        if v % FR_MODULUS != 0:
            raise ValueError("quotient degree too large")

    coeffs = list(t_coeff_full[: 3 * n])
    if len(coeffs) < 3 * n:
        coeffs.extend([0] * (3 * n - len(coeffs)))
    t1_coeff = tuple(coeffs[0:n])
    t2_coeff = tuple(coeffs[n : 2 * n])
    t3_coeff = tuple(coeffs[2 * n : 3 * n])
    return t1_coeff, t2_coeff, t3_coeff


def _collect_evals(pk: ProvingKey, a, b, c, z, t1, t2, t3, pi, zeta: int) -> Dict[str, int]:
    return {
        "a": poly_eval(a, zeta) % FR_MODULUS,
        "b": poly_eval(b, zeta) % FR_MODULUS,
        "c": poly_eval(c, zeta) % FR_MODULUS,
        "z": poly_eval(z, zeta) % FR_MODULUS,
        "t1": poly_eval(t1, zeta) % FR_MODULUS,
        "t2": poly_eval(t2, zeta) % FR_MODULUS,
        "t3": poly_eval(t3, zeta) % FR_MODULUS,
        "s1": poly_eval(pk.coeff_sigma1, zeta) % FR_MODULUS,
        "s2": poly_eval(pk.coeff_sigma2, zeta) % FR_MODULUS,
        "s3": poly_eval(pk.coeff_sigma3, zeta) % FR_MODULUS,
        "ql": poly_eval(pk.coeff_ql, zeta) % FR_MODULUS,
        "qr": poly_eval(pk.coeff_qr, zeta) % FR_MODULUS,
        "qm": poly_eval(pk.coeff_qm, zeta) % FR_MODULUS,
        "qo": poly_eval(pk.coeff_qo, zeta) % FR_MODULUS,
        "qc": poly_eval(pk.coeff_qc, zeta) % FR_MODULUS,
        "pi": poly_eval(pi, zeta) % FR_MODULUS,
    }


def _fold(polys, v: int) -> Tuple[Tuple[int, ...], G1, int]:
    v = int(v) % FR_MODULUS
    combined_coeff: List[int] = []
    combined_cm: G1 = G1_ZERO
    combined_y = 0
    power = 1
    for _, coeffs, cm, y in polys:
        if len(combined_coeff) < len(coeffs):
            combined_coeff.extend([0] * (len(coeffs) - len(combined_coeff)))
        for i, c in enumerate(coeffs):
            combined_coeff[i] = (combined_coeff[i] + power * (int(c) % FR_MODULUS)) % FR_MODULUS
        combined_cm = g1_add(combined_cm, g1_mul(cm, power))
        combined_y = (combined_y + power * (int(y) % FR_MODULUS)) % FR_MODULUS
        power = (power * v) % FR_MODULUS
    return tuple(combined_coeff), combined_cm, combined_y
