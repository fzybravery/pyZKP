from __future__ import annotations

from typing import Dict, List

from backend.schemes.groth16.r1cs import R1CSInstance, compile_r1cs
from backend.schemes.groth16.types import ProvingKey, VerifyingKey
from common.crypto.ecc.bn254 import G1, G2, G1_GENERATOR, G2_GENERATOR, g1_mul, g2_mul
from common.crypto.field.fr import FR_MODULUS, fr_inv, fr_rand
from common.crypto.poly import coeffs_from_evals_on_roots, omega_for_size, poly_eval
from common.ir.core import CircuitIR

"""
执行 Groth16 的可信设置 (Trusted Setup) 阶段。
工作流程：
1. 生成随机的陷门参数 (tau, alpha, beta, gamma, delta)。
2. 将 R1CS 矩阵的列插值为多项式 A_i(x), B_i(x), C_i(x)。
3. 在隐藏点 tau 处对多项式求值，并映射到椭圆曲线 G1/G2 上。
4. 生成供证明者计算 A, B 的 a_query, b_query。
5. 生成供验证者使用的公开输入查询 ic (用 gamma 隔离)。
6. 生成供证明者使用的私密输入查询 l_query (用 delta 隔离) 以及商多项式查询 h_query。
注意：在生产环境中，陷门参数 (trapdoor) 在执行完毕后必须被销毁。
"""


def setup(ir: CircuitIR) -> ProvingKey:
    r1cs = compile_r1cs(ir)
    n = r1cs.n_constraints
    m = r1cs.n_vars
    one_id = r1cs.one_id
    omega = omega_for_size(n)
    t_poly = [(-1) % FR_MODULUS] + [0] * (n - 1) + [1]

    while True:
        tau = fr_rand(nonzero=True)
        t_tau = (pow(tau, n, FR_MODULUS) - 1) % FR_MODULUS
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

        a_coeff = coeffs_from_evals_on_roots(ays, omega=omega)
        b_coeff = coeffs_from_evals_on_roots(bys, omega=omega)
        c_coeff = coeffs_from_evals_on_roots(cys, omega=omega)
        a_tau = poly_eval(a_coeff, tau)
        b_tau = poly_eval(b_coeff, tau)
        c_tau = poly_eval(c_coeff, tau)

        a_query.append(g1_mul(G1_GENERATOR, a_tau))
        b_query_g2.append(g2_mul(G2_GENERATOR, b_tau))
        b_query_g1.append(g1_mul(G1_GENERATOR, b_tau))

        k_tau = (beta * a_tau + alpha * b_tau + c_tau) % FR_MODULUS
        if var_id in pub_set:
            ic_map[var_id] = g1_mul(G1_GENERATOR, (k_tau * inv_gamma) % FR_MODULUS)
        else:
            l_map[var_id] = g1_mul(G1_GENERATOR, (k_tau * inv_delta) % FR_MODULUS)

    h_query: List[G1] = []
    t_tau = (pow(tau, n, FR_MODULUS) - 1) % FR_MODULUS
    for k in range(n - 1):
        s = (pow(tau, k, FR_MODULUS) * t_tau) % FR_MODULUS
        s = (s * inv_delta) % FR_MODULUS
        h_query.append(g1_mul(G1_GENERATOR, s))

    ic: List[G1] = []
    for pid in r1cs.public_ids:
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
