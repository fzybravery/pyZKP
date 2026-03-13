from __future__ import annotations

from typing import Dict, List

from pyZKP.backend.schemes.groth16.r1cs import R1CSInstance, compile_r1cs
from pyZKP.backend.schemes.groth16.types import ProvingKey, VerifyingKey
from pyZKP.common.crypto.ecc.bn254 import G1, G2, G1_GENERATOR, G2_GENERATOR, g1_mul, g2_mul
from pyZKP.common.crypto.field.fr import FR_MODULUS, fr_inv, fr_rand
from pyZKP.common.crypto.poly import barycentric_precompute, barycentric_value, poly_eval, poly_vanishing_from_roots
from pyZKP.common.ir.core import CircuitIR


def setup(ir: CircuitIR) -> ProvingKey:
    r1cs = compile_r1cs(ir)
    n = r1cs.n_constraints
    m = r1cs.n_vars
    one_id = r1cs.one_id

    xs = [(i + 1) % FR_MODULUS for i in range(n)]
    xs_pre, ws = barycentric_precompute(xs)

    t_poly = poly_vanishing_from_roots(xs)

    while True:
        tau = fr_rand(nonzero=True)
        t_tau = poly_eval(t_poly, tau)
        if t_tau % FR_MODULUS != 0:
            break

    alpha = fr_rand(nonzero=True)
    beta = fr_rand(nonzero=True)
    gamma = fr_rand(nonzero=True)
    delta = fr_rand(nonzero=True)

    inv_gamma = fr_inv(gamma)
    inv_delta = fr_inv(delta)

    alpha_g1 = g1_mul(G1_GENERATOR, alpha)
    beta_g1 = g1_mul(G1_GENERATOR, beta)
    beta_g2 = g2_mul(G2_GENERATOR, beta)
    gamma_g2 = g2_mul(G2_GENERATOR, gamma)
    delta_g1 = g1_mul(G1_GENERATOR, delta)
    delta_g2 = g2_mul(G2_GENERATOR, delta)

    a_query: List[G1] = []
    b_query_g2: List[G2] = []
    b_query_g1: List[G1] = []

    pub_set = set(r1cs.public_ids)
    aux_ids = list(r1cs.aux_ids)
    l_map: Dict[int, G1] = {}
    ic_map: Dict[int, G1] = {}

    for var_id in range(m):
        ays = [_row_value(r1cs.a_rows[j], var_id) for j in range(n)]
        bys = [_row_value(r1cs.b_rows[j], var_id) for j in range(n)]
        cys = [_row_value(r1cs.c_rows[j], var_id) for j in range(n)]

        a_tau = barycentric_value(xs_pre, ws, ays, tau)
        b_tau = barycentric_value(xs_pre, ws, bys, tau)
        c_tau = barycentric_value(xs_pre, ws, cys, tau)

        a_query.append(g1_mul(G1_GENERATOR, a_tau))
        b_query_g2.append(g2_mul(G2_GENERATOR, b_tau))
        b_query_g1.append(g1_mul(G1_GENERATOR, b_tau))

        k_tau = (beta * a_tau + alpha * b_tau + c_tau) % FR_MODULUS
        if var_id in pub_set:
            ic_map[var_id] = g1_mul(G1_GENERATOR, (k_tau * inv_gamma) % FR_MODULUS)
        else:
            l_map[var_id] = g1_mul(G1_GENERATOR, (k_tau * inv_delta) % FR_MODULUS)

    h_query: List[G1] = []
    t_tau = poly_eval(t_poly, tau) % FR_MODULUS
    for k in range(n - 1):
        s = (pow(tau, k, FR_MODULUS) * t_tau) % FR_MODULUS
        s = (s * inv_delta) % FR_MODULUS
        h_query.append(g1_mul(G1_GENERATOR, s))

    ic: List[G1] = []
    for pid in r1cs.public_ids:
        if pid == one_id:
            ays = [_row_value(r1cs.a_rows[j], one_id) for j in range(n)]
            bys = [_row_value(r1cs.b_rows[j], one_id) for j in range(n)]
            cys = [_row_value(r1cs.c_rows[j], one_id) for j in range(n)]
            a_tau = barycentric_value(xs_pre, ws, ays, tau)
            b_tau = barycentric_value(xs_pre, ws, bys, tau)
            c_tau = barycentric_value(xs_pre, ws, cys, tau)
            k_tau = (beta * a_tau + alpha * b_tau + c_tau) % FR_MODULUS
            ic.append(g1_mul(G1_GENERATOR, (k_tau * inv_gamma) % FR_MODULUS))
        else:
            ic.append(ic_map[pid])

    l_query: List[G1] = [l_map[i] for i in aux_ids]

    vk = VerifyingKey(
        alpha_g1=alpha_g1,
        beta_g2=beta_g2,
        gamma_g2=gamma_g2,
        delta_g2=delta_g2,
        ic=ic,
        public_names=r1cs.public_names,
    )
    return ProvingKey(
        vk=vk,
        beta_g1=beta_g1,
        delta_g1=delta_g1,
        a_query=a_query,
        b_query_g2=b_query_g2,
        b_query_g1=b_query_g1,
        h_query=h_query,
        l_query=l_query,
        aux_ids=aux_ids,
        trapdoor_tau=tau,
        trapdoor_alpha=alpha,
        trapdoor_beta=beta,
        trapdoor_gamma=gamma,
        trapdoor_delta=delta,
    )


def _row_value(row: Dict[int, int], var_id: int) -> int:
    return int(row.get(var_id, 0)) % FR_MODULUS
