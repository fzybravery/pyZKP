from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from backend.schemes.plonk.transcript import Transcript
from backend.schemes.plonk.types import Proof, VerifyingKey
from common.crypto.ecc.bn254 import G1, G1_GENERATOR, G1_ZERO, g1_add, g1_mul, g1_sub
from common.crypto.field.fr import FR_MODULUS, fr_inv
from common.crypto.poly import barycentric_precompute, barycentric_value, poly_eval


def verify(vk: VerifyingKey, proof: Proof, public_values: Sequence[int]) -> bool:
    n = vk.domain.n
    omega = vk.domain.omega
    roots = list(vk.domain.roots)

    if len(public_values) != 1 + (len(vk.public_names) - 1):
        return False
    if int(public_values[0]) % FR_MODULUS != 1:
        return False

    tr = Transcript()
    tr.absorb_g1(vk.cm_sigma1)
    tr.absorb_g1(vk.cm_sigma2)
    tr.absorb_g1(vk.cm_sigma3)
    tr.absorb_g1(vk.cm_ql)
    tr.absorb_g1(vk.cm_qr)
    tr.absorb_g1(vk.cm_qm)
    tr.absorb_g1(vk.cm_qo)
    tr.absorb_g1(vk.cm_qc)
    for x in public_values:
        tr.absorb_int(int(x))
    tr.absorb_g1(proof.cm_a)
    tr.absorb_g1(proof.cm_b)
    tr.absorb_g1(proof.cm_c)
    beta = tr.challenge(b"beta")
    gamma = tr.challenge(b"gamma")
    tr.absorb_g1(proof.cm_z)
    alpha = tr.challenge(b"alpha")
    tr.absorb_g1(proof.cm_t1)
    tr.absorb_g1(proof.cm_t2)
    tr.absorb_g1(proof.cm_t3)
    zeta = tr.challenge(b"zeta")
    zeta_omega = (zeta * omega) % FR_MODULUS

    evals = proof.evals_zeta
    required = ["a", "b", "c", "z", "t1", "t2", "t3", "s1", "s2", "s3", "ql", "qr", "qm", "qo", "qc", "pi"]
    for k in required:
        if k not in evals:
            return False

    xs_pre, ws = barycentric_precompute(roots)
    pi_eval = [0] * n
    for i, pv in enumerate(public_values[1:]):
        row = 1 + i
        pi_eval[row] = (-int(pv)) % FR_MODULUS
    pi_z = barycentric_value(xs_pre, ws, pi_eval, zeta)
    if pi_z % FR_MODULUS != int(evals["pi"]) % FR_MODULUS:
        return False

    zh = (pow(zeta, n, FR_MODULUS) - 1) % FR_MODULUS
    if zh == 0:
        return False

    l1 = (zh * fr_inv((n * ((zeta - 1) % FR_MODULUS)) % FR_MODULUS)) % FR_MODULUS

    a = int(evals["a"]) % FR_MODULUS
    b = int(evals["b"]) % FR_MODULUS
    c = int(evals["c"]) % FR_MODULUS
    z = int(evals["z"]) % FR_MODULUS
    z_w = int(proof.eval_zeta_omega) % FR_MODULUS
    t1_z = int(evals["t1"]) % FR_MODULUS
    t2_z = int(evals["t2"]) % FR_MODULUS
    t3_z = int(evals["t3"]) % FR_MODULUS
    zeta_n = pow(int(zeta) % FR_MODULUS, n, FR_MODULUS)
    t = (t1_z + zeta_n * t2_z + (zeta_n * zeta_n % FR_MODULUS) * t3_z) % FR_MODULUS
    s1 = int(evals["s1"]) % FR_MODULUS
    s2 = int(evals["s2"]) % FR_MODULUS
    s3 = int(evals["s3"]) % FR_MODULUS
    ql = int(evals["ql"]) % FR_MODULUS
    qr = int(evals["qr"]) % FR_MODULUS
    qm = int(evals["qm"]) % FR_MODULUS
    qo = int(evals["qo"]) % FR_MODULUS
    qc = int(evals["qc"]) % FR_MODULUS
    piv = int(evals["pi"]) % FR_MODULUS

    gate = (ql * a + qr * b + qm * a * b + qo * c + qc + piv) % FR_MODULUS

    r1 = (a + beta * zeta + gamma) % FR_MODULUS
    r2 = (b + beta * vk.k1 * zeta + gamma) % FR_MODULUS
    r3 = (c + beta * vk.k2 * zeta + gamma) % FR_MODULUS
    left = (((r1 * r2) % FR_MODULUS) * r3) % FR_MODULUS
    left = (left * z) % FR_MODULUS

    u1 = (a + beta * s1 + gamma) % FR_MODULUS
    u2 = (b + beta * s2 + gamma) % FR_MODULUS
    u3 = (c + beta * s3 + gamma) % FR_MODULUS
    right = (((u1 * u2) % FR_MODULUS) * u3) % FR_MODULUS
    right = (right * z_w) % FR_MODULUS
    perm = (left - right) % FR_MODULUS

    b1 = ((z - 1) % FR_MODULUS) * l1 % FR_MODULUS

    rhs = (gate + alpha * perm + (alpha * alpha % FR_MODULUS) * b1) % FR_MODULUS
    lhs = (t * zh) % FR_MODULUS
    if lhs != rhs:
        return False

    tr.absorb_int(a)
    tr.absorb_int(b)
    tr.absorb_int(c)
    tr.absorb_int(z)
    tr.absorb_int(z_w)
    tr.absorb_int(t1_z)
    tr.absorb_int(t2_z)
    tr.absorb_int(t3_z)
    tr.absorb_int(s1)
    tr.absorb_int(s2)
    tr.absorb_int(s3)
    tr.absorb_int(ql)
    tr.absorb_int(qr)
    tr.absorb_int(qm)
    tr.absorb_int(qo)
    tr.absorb_int(qc)
    v = tr.challenge(b"v")

    ok1 = _verify_batch_opening(vk, proof, zeta, v)
    if not ok1:
        return False
    ok2 = _verify_kzg(vk, proof.cm_z, zeta_omega, z_w, proof.pi_zeta_omega)
    return ok2


def _verify_batch_opening(vk: VerifyingKey, proof: Proof, zeta: int, v: int) -> bool:
    evals = proof.evals_zeta
    polys = [
        ("a", proof.cm_a, evals["a"]),
        ("b", proof.cm_b, evals["b"]),
        ("c", proof.cm_c, evals["c"]),
        ("z", proof.cm_z, evals["z"]),
        ("t1", proof.cm_t1, evals["t1"]),
        ("t2", proof.cm_t2, evals["t2"]),
        ("t3", proof.cm_t3, evals["t3"]),
        ("s1", vk.cm_sigma1, evals["s1"]),
        ("s2", vk.cm_sigma2, evals["s2"]),
        ("s3", vk.cm_sigma3, evals["s3"]),
        ("ql", vk.cm_ql, evals["ql"]),
        ("qr", vk.cm_qr, evals["qr"]),
        ("qm", vk.cm_qm, evals["qm"]),
        ("qo", vk.cm_qo, evals["qo"]),
        ("qc", vk.cm_qc, evals["qc"]),
    ]
    v = int(v) % FR_MODULUS
    power = 1
    cm: G1 = G1_ZERO
    y = 0
    for _, cmi, yi in polys:
        cm = g1_add(cm, g1_mul(cmi, power))
        y = (y + power * (int(yi) % FR_MODULUS)) % FR_MODULUS
        power = (power * v) % FR_MODULUS
    return _verify_kzg(vk, cm, zeta, y, proof.pi_zeta)


def _verify_kzg(vk: VerifyingKey, commitment: G1, z: int, y: int, proof: G1) -> bool:
    from py_ecc import optimized_bn128 as b

    zz = int(z) % FR_MODULUS
    yy = int(y) % FR_MODULUS
    g2_tau = vk.srs_tau_g2
    g2_0 = vk.srs_g2
    q = b.add(g2_tau, b.multiply(g2_0, (-zz) % FR_MODULUS))
    left = b.pairing(q, proof)
    right = b.pairing(g2_0, g1_sub(commitment, g1_mul(G1_GENERATOR, yy)))
    return left == right
