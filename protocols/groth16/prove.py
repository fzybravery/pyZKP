from __future__ import annotations

from typing import Any, Dict, List, Sequence

from protocols.groth16.qap import compute_h_from_abc_on_roots
from protocols.groth16.r1cs import compile_r1cs, eval_r1cs_vectors
from protocols.groth16.types import Proof, ProvingKey
from crypto.ecc.bn254 import G1, G1_ZERO, G2, g1_add, g1_mul, g1_sub, g2_add, g2_mul
from crypto.field.fr import FR_MODULUS, fr_rand
from crypto.poly import omega_for_size
from frontend.ir.core import CircuitIR
from frontend.api.witness import Witness
from runtime import Executor, KernelRegistry, RuntimeConfig
from runtime.ir import Device, DType, Graph, OpType
from runtime.kernels.cpu import register_cpu_kernels


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
def prove(
    ir: CircuitIR,
    pk: ProvingKey,
    witness: Witness,
    *,
    runtime_trace=None,
    runtime_pool=None,
    runtime_context=None,
    runtime_config: RuntimeConfig | None = None,
    runtime_attrs: Dict[str, Any] | None = None,
) -> Proof:
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
    from runtime.ir import Backend

    backend0 = runtime_config.backend if runtime_config is not None else Backend.CPU
    if runtime_context is not None:
        backend0 = runtime_context.backend
    register_cpu_kernels(reg, backend=backend0)
    if backend0 == Backend.METAL:
        from runtime.kernels.metal import register_metal_kernels
        register_metal_kernels(reg)
    exe = Executor(registry=reg)
    attrs0: Dict[str, Any] = runtime_config.with_overrides(runtime_attrs) if runtime_config is not None else dict(runtime_attrs or {})
    from runtime.warmup import apply_fixed_base_policy_groth16

    attrs0 = apply_fixed_base_policy_groth16(pk, attrs0)
    ctx0 = runtime_config.make_context(pool=runtime_pool, context=runtime_context) if runtime_config is not None else runtime_context
    g = Graph()

    # 将数据存储到runtime缓冲区中
    g.add_buffer(id="w", device=Device.CPU, dtype=DType.FR, data=w)
    from runtime.warmup import cached_points_tuple

    g.add_buffer(id="a_query", device=Device.CPU, dtype=DType.G1, data=cached_points_tuple(pk.a_query))
    g.add_buffer(id="b_query_g2", device=Device.CPU, dtype=DType.G2, data=list(pk.b_query_g2))
    g.add_buffer(id="b_query_g1", device=Device.CPU, dtype=DType.G1, data=cached_points_tuple(pk.b_query_g1))

    # 添加计算节点
    g.add_node(op=OpType.MSM_G1, inputs=["a_query", "w"], outputs=["a_lin"], attrs=attrs0)
    g.add_node(op=OpType.MSM_G2, inputs=["b_query_g2", "w"], outputs=["b_lin_g2"], attrs=attrs0)
    g.add_node(op=OpType.MSM_G1, inputs=["b_query_g1", "w"], outputs=["b_lin_g1"], attrs=attrs0)

    r = fr_rand(nonzero=True)
    s = fr_rand(nonzero=True)

    # 计算并且获取结果
    exe.run(g, pool=runtime_pool, trace=runtime_trace, keep=["a_lin", "b_lin_g2", "b_lin_g1"], context=ctx0)
    a_lin: G1 = g.buffers["a_lin"].data
    b_lin_g2: G2 = g.buffers["b_lin_g2"].data
    b_lin_g1: G1 = g.buffers["b_lin_g1"].data

    a = g1_add(pk.vk.alpha_g1, g1_add(a_lin, g1_mul(pk.delta_g1, r)))
    b = g2_add(pk.vk.beta_g2, g2_add(b_lin_g2, g2_mul(pk.vk.delta_g2, s)))

    a_eval, b_eval, c_eval = eval_r1cs_vectors(r1cs, w)
    omega = omega_for_size(n)
    qap = compute_h_from_abc_on_roots(n, omega, a_eval, b_eval, c_eval, runtime_trace=runtime_trace, runtime_pool=runtime_pool)
    h_coeffs = list(qap.h_poly)
    if len(h_coeffs) < n - 1:
        h_coeffs = h_coeffs + [0] * (n - 1 - len(h_coeffs))
    if len(h_coeffs) > n - 1:
        h_coeffs = h_coeffs[: n - 1]

    aux_scalars = [w[i] for i in pk.aux_ids]

    g2 = Graph()
    g2.add_buffer(id="h_query", device=Device.CPU, dtype=DType.G1, data=cached_points_tuple(pk.h_query))
    g2.add_buffer(id="h_scalars", device=Device.CPU, dtype=DType.FR, data=h_coeffs)
    g2.add_node(op=OpType.MSM_G1, inputs=["h_query", "h_scalars"], outputs=["h_acc"], attrs=attrs0)

    if len(pk.l_query) != 0:
        g2.add_buffer(id="l_query", device=Device.CPU, dtype=DType.G1, data=cached_points_tuple(pk.l_query))
        g2.add_buffer(id="l_scalars", device=Device.CPU, dtype=DType.FR, data=aux_scalars)
        g2.add_node(op=OpType.MSM_G1, inputs=["l_query", "l_scalars"], outputs=["l_acc"], attrs=attrs0)

    exe.run(g2, pool=runtime_pool, trace=runtime_trace, keep=["h_acc", "l_acc"], context=ctx0)
    h_acc: G1 = g2.buffers["h_acc"].data
    l_acc: G1 = g2.buffers["l_acc"].data if "l_acc" in g2.buffers else G1_ZERO

    sa = g1_mul(g1_add(pk.vk.alpha_g1, a_lin), s)
    rb = g1_mul(g1_add(pk.beta_g1, b_lin_g1), r)
    rs = (r * s) % FR_MODULUS
    rs_delta = g1_mul(pk.delta_g1, rs)

    c = g1_add(g1_add(l_acc, h_acc), g1_add(g1_add(sa, rb), rs_delta))

    return Proof(a=a, b=b, c=c)

# 批量证明，对于一个 pk，对多个 witness 连续生成多个证明
# 并且整个过程，只编译一次 R1CS 约束，只注册一次 kernel，只创建一次 executor
# 两段MSM的图只构建一次并复用，复用的是证明里用到的 MSM 图
def prove_batch(
    ir: CircuitIR,
    pk: ProvingKey,
    witnesses: Sequence[Witness],
    *,
    runtime_trace=None,
    runtime_pool=None,
    runtime_context=None,
    runtime_config: RuntimeConfig | None = None,
    runtime_attrs: Dict[str, Any] | None = None,
) -> List[Proof]:
    r1cs = compile_r1cs(ir)
    n = r1cs.n_constraints
    one_id = r1cs.one_id
    omega = omega_for_size(n)

    reg = KernelRegistry()
    from runtime.ir import Backend

    backend0 = runtime_config.backend if runtime_config is not None else Backend.CPU
    if runtime_context is not None:
        backend0 = runtime_context.backend
    register_cpu_kernels(reg, backend=backend0)
    if backend0 == Backend.METAL:
        from runtime.kernels.metal import register_metal_kernels
        register_metal_kernels(reg)
    exe = Executor(registry=reg)
    attrs0: Dict[str, Any] = runtime_config.with_overrides(runtime_attrs) if runtime_config is not None else dict(runtime_attrs or {})
    from runtime.warmup import apply_fixed_base_policy_groth16, cached_points_tuple

    attrs0 = apply_fixed_base_policy_groth16(pk, attrs0)
    ctx0 = runtime_config.make_context(pool=runtime_pool, context=runtime_context) if runtime_config is not None else runtime_context

    a_query = cached_points_tuple(pk.a_query)
    b_query_g2 = list(pk.b_query_g2)
    b_query_g1 = cached_points_tuple(pk.b_query_g1)
    h_query = cached_points_tuple(pk.h_query)
    l_query = cached_points_tuple(pk.l_query) if len(pk.l_query) != 0 else None

    g1 = Graph()
    g1.add_buffer(id="w", device=Device.CPU, dtype=DType.FR, data=[0] * (one_id + 1))
    g1.add_buffer(id="a_query", device=Device.CPU, dtype=DType.G1, data=a_query)
    g1.add_buffer(id="b_query_g2", device=Device.CPU, dtype=DType.G2, data=b_query_g2)
    g1.add_buffer(id="b_query_g1", device=Device.CPU, dtype=DType.G1, data=b_query_g1)
    g1.add_node(op=OpType.MSM_G1, inputs=["a_query", "w"], outputs=["a_lin"], attrs=attrs0)
    g1.add_node(op=OpType.MSM_G2, inputs=["b_query_g2", "w"], outputs=["b_lin_g2"], attrs=attrs0)
    g1.add_node(op=OpType.MSM_G1, inputs=["b_query_g1", "w"], outputs=["b_lin_g1"], attrs=attrs0)
    a1 = g1.analyze_cached()
    keep1 = list(a1.initial) + ["a_lin", "b_lin_g2", "b_lin_g1"]

    g2 = Graph()
    g2.add_buffer(id="h_query", device=Device.CPU, dtype=DType.G1, data=h_query)
    g2.add_buffer(id="h_scalars", device=Device.CPU, dtype=DType.FR, data=[0] * (n - 1))
    g2.add_node(op=OpType.MSM_G1, inputs=["h_query", "h_scalars"], outputs=["h_acc"], attrs=attrs0)
    if l_query is not None:
        g2.add_buffer(id="l_query", device=Device.CPU, dtype=DType.G1, data=l_query)
        g2.add_buffer(id="l_scalars", device=Device.CPU, dtype=DType.FR, data=[0] * len(pk.aux_ids))
        g2.add_node(op=OpType.MSM_G1, inputs=["l_query", "l_scalars"], outputs=["l_acc"], attrs=attrs0)
    a2 = g2.analyze_cached()
    keep2 = list(a2.initial) + ["h_acc", "l_acc"]

    proofs: List[Proof] = []
    for i, wit in enumerate(witnesses):
        w = list(wit.values)
        if len(w) != one_id:
            raise ValueError("witness length mismatch")
        w.append(1)
        g1.buffers["w"].data[:] = w

        r = fr_rand(nonzero=True)
        s = fr_rand(nonzero=True)

        exe.run(g1, pool=runtime_pool, trace=runtime_trace, keep=keep1, context=ctx0)
        a_lin: G1 = g1.buffers["a_lin"].data
        b_lin_g2: G2 = g1.buffers["b_lin_g2"].data
        b_lin_g1: G1 = g1.buffers["b_lin_g1"].data

        a = g1_add(pk.vk.alpha_g1, g1_add(a_lin, g1_mul(pk.delta_g1, r)))
        b = g2_add(pk.vk.beta_g2, g2_add(b_lin_g2, g2_mul(pk.vk.delta_g2, s)))

        a_eval, b_eval, c_eval = eval_r1cs_vectors(r1cs, w)
        qap = compute_h_from_abc_on_roots(
            n,
            omega,
            a_eval,
            b_eval,
            c_eval,
            runtime_trace=runtime_trace,
            runtime_pool=runtime_pool,
            runtime_context=ctx0,
            runtime_config=runtime_config,
        )
        h_coeffs = list(qap.h_poly)
        if len(h_coeffs) < n - 1:
            h_coeffs = h_coeffs + [0] * (n - 1 - len(h_coeffs))
        if len(h_coeffs) > n - 1:
            h_coeffs = h_coeffs[: n - 1]

        g2.buffers["h_scalars"].data[:] = h_coeffs
        if l_query is not None:
            aux_scalars = [w[j] for j in pk.aux_ids]
            g2.buffers["l_scalars"].data[:] = aux_scalars

        exe.run(g2, pool=runtime_pool, trace=runtime_trace, keep=keep2, context=ctx0)
        h_acc: G1 = g2.buffers["h_acc"].data
        l_acc: G1 = g2.buffers["l_acc"].data if "l_acc" in g2.buffers else G1_ZERO

        sa = g1_mul(g1_add(pk.vk.alpha_g1, a_lin), s)
        rb = g1_mul(g1_add(pk.beta_g1, b_lin_g1), r)
        rs = (r * s) % FR_MODULUS
        rs_delta = g1_mul(pk.delta_g1, rs)
        c = g1_add(g1_add(l_acc, h_acc), g1_add(g1_add(sa, rb), rs_delta))

        proofs.append(Proof(a=a, b=b, c=c))

    return proofs
