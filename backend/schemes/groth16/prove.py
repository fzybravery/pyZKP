from __future__ import annotations

from typing import List

from pyZKP.backend.schemes.groth16.qap import compute_h_from_abc_on_roots
from pyZKP.backend.schemes.groth16.r1cs import compile_r1cs, eval_r1cs_vectors
from pyZKP.backend.schemes.groth16.types import Proof, ProvingKey
from pyZKP.common.crypto.ecc.bn254 import G1, G1_ZERO, G2, g1_add, g1_mul, g1_sub, g2_add, g2_mul
from pyZKP.common.crypto.field.fr import FR_MODULUS, fr_rand
from pyZKP.common.crypto.poly import omega_for_size
from pyZKP.common.ir.core import CircuitIR
from pyZKP.frontend.api.witness import Witness
from pyZKP.runtime import Executor, KernelRegistry
from pyZKP.runtime.ir import Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels


"""
基于 Runtime 执行引擎生成 Groth16 零知识证明。
参数:
    ir: 电路中间表示 (CircuitIR)
    pk: 证明密钥 (ProvingKey)
    witness: 包含所有变量取值的见证表
返回:
    包含 A, B, C 三个椭圆曲线点的 Proof 对象。
说明:
    该实现利用 Graph 和 Executor 将底层的多标量乘法 (MSM) 任务抽象为计算图节点，
    实现了前端逻辑与底层硬件执行的解耦，为后续的 GPU 加速铺平了道路。
"""
def prove(ir: CircuitIR, pk: ProvingKey, witness: Witness) -> Proof:
    # 编译 R1CS 约束
    r1cs = compile_r1cs(ir)
    n = r1cs.n_constraints
    one_id = r1cs.one_id

    # 获取 witness 列表，并且在末尾补 1
    w = list(witness.values)
    if len(w) != one_id:
        raise ValueError("witness length mismatch")
    w.append(1)

    # 初始化 Runtime 执行引擎
    reg = KernelRegistry()
    register_cpu_kernels(reg)
    exe = Executor(registry=reg)
    g = Graph()

    # 将数据存储到runtime缓冲区中
    g.add_buffer(id="w", device=Device.CPU, dtype=DType.FR, data=w)
    g.add_buffer(id="a_query", device=Device.CPU, dtype=DType.G1, data=list(pk.a_query))
    g.add_buffer(id="b_query_g2", device=Device.CPU, dtype=DType.G2, data=list(pk.b_query_g2))
    g.add_buffer(id="b_query_g1", device=Device.CPU, dtype=DType.G1, data=list(pk.b_query_g1))

    # 添加计算节点
    g.add_node(op=OpType.MSM_G1, inputs=["a_query", "w"], outputs=["a_lin"])
    g.add_node(op=OpType.MSM_G2, inputs=["b_query_g2", "w"], outputs=["b_lin_g2"])
    g.add_node(op=OpType.MSM_G1, inputs=["b_query_g1", "w"], outputs=["b_lin_g1"])

    r = fr_rand(nonzero=True)
    s = fr_rand(nonzero=True)

    # 计算并且获取结果
    exe.run(g)
    a_lin: G1 = g.buffers["a_lin"].data
    b_lin_g2: G2 = g.buffers["b_lin_g2"].data
    b_lin_g1: G1 = g.buffers["b_lin_g1"].data

    a = g1_add(pk.vk.alpha_g1, g1_add(a_lin, g1_mul(pk.delta_g1, r)))
    b = g2_add(pk.vk.beta_g2, g2_add(b_lin_g2, g2_mul(pk.vk.delta_g2, s)))

    a_eval, b_eval, c_eval = eval_r1cs_vectors(r1cs, w)
    omega = omega_for_size(n)
    qap = compute_h_from_abc_on_roots(n, omega, a_eval, b_eval, c_eval)
    h_coeffs = list(qap.h_poly)
    if len(h_coeffs) < n - 1:
        h_coeffs = h_coeffs + [0] * (n - 1 - len(h_coeffs))
    if len(h_coeffs) > n - 1:
        h_coeffs = h_coeffs[: n - 1]

    aux_scalars = [w[i] for i in pk.aux_ids]

    g2 = Graph()
    g2.add_buffer(id="h_query", device=Device.CPU, dtype=DType.G1, data=list(pk.h_query))
    g2.add_buffer(id="h_scalars", device=Device.CPU, dtype=DType.FR, data=h_coeffs)
    g2.add_node(op=OpType.MSM_G1, inputs=["h_query", "h_scalars"], outputs=["h_acc"])

    if len(pk.l_query) != 0:
        g2.add_buffer(id="l_query", device=Device.CPU, dtype=DType.G1, data=list(pk.l_query))
        g2.add_buffer(id="l_scalars", device=Device.CPU, dtype=DType.FR, data=aux_scalars)
        g2.add_node(op=OpType.MSM_G1, inputs=["l_query", "l_scalars"], outputs=["l_acc"])

    exe.run(g2)
    h_acc: G1 = g2.buffers["h_acc"].data
    l_acc: G1 = g2.buffers["l_acc"].data if "l_acc" in g2.buffers else G1_ZERO

    sa = g1_mul(g1_add(pk.vk.alpha_g1, a_lin), s)
    rb = g1_mul(g1_add(pk.beta_g1, b_lin_g1), r)
    rs = (r * s) % FR_MODULUS
    rs_delta = g1_mul(pk.delta_g1, rs)

    c = g1_add(g1_add(l_acc, h_acc), g1_add(g1_add(sa, rb), rs_delta))

    return Proof(a=a, b=b, c=c)
