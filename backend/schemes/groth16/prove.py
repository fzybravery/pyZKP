from __future__ import annotations

from typing import List

from pyZKP.backend.schemes.groth16.qap import compute_h_from_abc
from pyZKP.backend.schemes.groth16.r1cs import compile_r1cs, eval_r1cs_vectors
from pyZKP.backend.schemes.groth16.types import Proof, ProvingKey
from pyZKP.common.crypto.ecc.bn254 import G1, G1_ZERO, G2, g1_add, g1_mul, g1_sub, g2_add, g2_mul
from pyZKP.common.crypto.field.fr import FR_MODULUS, fr_rand
from pyZKP.common.crypto.msm import msm_naive_g1, msm_naive_g2
from pyZKP.common.ir.core import CircuitIR
from pyZKP.frontend.api.witness import Witness


def prove(ir: CircuitIR, pk: ProvingKey, witness: Witness) -> Proof:
    r1cs = compile_r1cs(ir)
    n = r1cs.n_constraints
    one_id = r1cs.one_id

    w = list(witness.values)
    if len(w) != one_id:
        raise ValueError("witness length mismatch")
    w.append(1)

    a_lin = msm_naive_g1(pk.a_query, w)
    b_lin_g2 = msm_naive_g2(pk.b_query_g2, w)
    b_lin_g1 = msm_naive_g1(pk.b_query_g1, w)

    r = fr_rand(nonzero=True)
    s = fr_rand(nonzero=True)

    a = g1_add(pk.vk.alpha_g1, g1_add(a_lin, g1_mul(pk.delta_g1, r)))
    b = g2_add(pk.vk.beta_g2, g2_add(b_lin_g2, g2_mul(pk.vk.delta_g2, s)))

    a_eval, b_eval, c_eval = eval_r1cs_vectors(r1cs, w)
    xs = [(i + 1) % FR_MODULUS for i in range(n)]
    qap = compute_h_from_abc(xs, a_eval, b_eval, c_eval)
    h_coeffs = list(qap.h_poly)
    if len(h_coeffs) < n - 1:
        h_coeffs = h_coeffs + [0] * (n - 1 - len(h_coeffs))
    if len(h_coeffs) > n - 1:
        h_coeffs = h_coeffs[: n - 1]

    h_acc = msm_naive_g1(pk.h_query, h_coeffs)

    aux_scalars = [w[i] for i in pk.aux_ids]
    l_acc = msm_naive_g1(pk.l_query, aux_scalars) if len(pk.l_query) != 0 else G1_ZERO

    sa = g1_mul(g1_add(pk.vk.alpha_g1, a_lin), s)
    rb = g1_mul(g1_add(pk.beta_g1, b_lin_g1), r)
    rs = (r * s) % FR_MODULUS
    rs_delta = g1_mul(pk.delta_g1, rs)

    c = g1_add(g1_add(l_acc, h_acc), g1_add(g1_add(sa, rb), rs_delta))

    return Proof(a=a, b=b, c=c)
