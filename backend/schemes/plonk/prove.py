from __future__ import annotations

import secrets
from typing import Any, Dict, List, Sequence, Tuple

from pyZKP.backend.schemes.plonk.transcript import Transcript
from pyZKP.backend.schemes.plonk.types import Proof, ProvingKey
from pyZKP.common.crypto.ecc.bn254 import G1, G1_ZERO, g1_add, g1_mul
from pyZKP.common.crypto.field.fr import FR_MODULUS, fr_inv
from pyZKP.common.crypto.kzg.cpu_ref import commit
from pyZKP.common.crypto.poly import (
    coeffs_from_evals_on_roots,
    omega_for_size,
    poly_eval,
)
from pyZKP.frontend.api.witness import Witness
from pyZKP.runtime import Executor, KernelRegistry, RuntimeConfig
from pyZKP.runtime.ir import Backend, Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels


# 核心证明生成函数（Prover）。
# 严格按照 PLONK 协议流程执行：
# 1. 补全所有门电路导线的求值 (a, b, c)。
# 2. 生成线缆多项式的 KZG 承诺。
# 3. 通过 Transcript 获取挑战因子 beta, gamma，计算并承诺置换多项式 Z。
# 4. 获取 alpha，在扩展的陪集域上计算商多项式 T，将其分割为 t1, t2, t3 并承诺。
# 5. 获取求值点 zeta，计算所有多项式在 zeta (及 zeta*omega) 处的求值。
# 6. 利用挑战因子 v 将所有多项式折叠（Fold），生成批量的 KZG 打开证明（pi_zeta, pi_zeta_omega）。

def prove(
    pk: ProvingKey,
    witness: Witness,
    public_values: Sequence[int],
    *,
    runtime_trace=None,
    runtime_pool=None,
    runtime_context=None,
    runtime_config: RuntimeConfig | None = None,
    runtime_attrs: Dict[str, Any] | None = None,
) -> Proof:
    c = pk.circuit
    n = c.domain.n
    omega = c.domain.omega

    if len(public_values) != 1 + len(c.public_var_ids):
        raise ValueError("public_values must be [ONE] + public inputs in schema order")
    if int(public_values[0]) % FR_MODULUS != 1:
        raise ValueError("public_values[0] must be ONE==1")

    # 将 witness 中的变量值填入门电路
    values = _build_extended_values(c, witness)
    a_eval = [values[g.l] for g in c.gates]
    b_eval = [values[g.r] for g in c.gates]
    c_eval = [values[g.o] for g in c.gates]
    
    # 将线性组合的系数转换为多项式系数
    a_coeff = tuple(coeffs_from_evals_on_roots(a_eval, omega=omega))
    b_coeff = tuple(coeffs_from_evals_on_roots(b_eval, omega=omega))
    c_coeff = tuple(coeffs_from_evals_on_roots(c_eval, omega=omega))

    reg = KernelRegistry()
    backend0 = runtime_config.backend if runtime_config is not None else Backend.CPU
    if runtime_context is not None:
        backend0 = runtime_context.backend
    register_cpu_kernels(reg, backend=backend0)
    if backend0 == Backend.METAL:
        from pyZKP.runtime.kernels.metal import register_metal_kernels
        register_metal_kernels(reg)
    exe = Executor(registry=reg)
    pool = runtime_pool
    trace = runtime_trace
    ctx0 = runtime_config.make_context(pool=pool, context=runtime_context) if runtime_config is not None else runtime_context
    attrs0: Dict[str, Any] = runtime_config.with_overrides(runtime_attrs) if runtime_config is not None else dict(runtime_attrs or {})
    from pyZKP.runtime.warmup import apply_fixed_base_policy_plonk

    attrs0 = apply_fixed_base_policy_plonk(pk, attrs0)

    # 生成 A, B, C 的 KZG 证明
    g0 = Graph()
    g0.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=pk.srs)
    g0.add_buffer(id="abc_coeffs", device=Device.CPU, dtype=DType.OBJ, data=[list(a_coeff), list(b_coeff), list(c_coeff)])
    g0.add_node(op=OpType.KZG_BATCH_COMMIT, inputs=["srs", "abc_coeffs"], outputs=["abc_cms"], attrs=attrs0)
    exe.run(g0, pool=pool, trace=trace, keep=["abc_cms"], context=ctx0)
    cm_a, cm_b, cm_c = g0.buffers["abc_cms"].data

    pi_eval = [0] * n
    for i, pv in enumerate(public_values[1:]):
        row = 1 + i
        pi_eval[row] = (-int(pv)) % FR_MODULUS
    pi_coeff = tuple(coeffs_from_evals_on_roots(pi_eval, omega=omega))

    # 通过挑战响应协议生成 beta 和 gamma
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

    # 生成并承诺置换多项式
    z_eval = _build_permutation_z(c, a_eval, b_eval, c_eval, beta, gamma)
    z_coeff = tuple(coeffs_from_evals_on_roots(z_eval, omega=omega))
    gz = Graph()
    gz.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=pk.srs)
    gz.add_buffer(id="z_coeffs", device=Device.CPU, dtype=DType.OBJ, data=[list(z_coeff)])
    gz.add_node(op=OpType.KZG_BATCH_COMMIT, inputs=["srs", "z_coeffs"], outputs=["z_cms"], attrs=attrs0)
    exe.run(gz, pool=pool, trace=trace, keep=["z_cms"], context=ctx0)
    cm_z = gz.buffers["z_cms"].data[0]

    # 生成 alpha
    tr.absorb_g1(cm_z)
    alpha = tr.challenge(b"alpha")
    
    # 在陪集上计算商多项式，并且分割为三部分，生成对应的承诺
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
        runtime_trace=runtime_trace,
        runtime_pool=runtime_pool,
        runtime_context=ctx0,
        runtime_config=runtime_config,
    )
    gt = Graph()
    gt.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=pk.srs)
    gt.add_buffer(id="t_coeffs", device=Device.CPU, dtype=DType.OBJ, data=[list(t1_coeff), list(t2_coeff), list(t3_coeff)])
    gt.add_node(op=OpType.KZG_BATCH_COMMIT, inputs=["srs", "t_coeffs"], outputs=["t_cms"], attrs=attrs0)
    exe.run(gt, pool=pool, trace=trace, keep=["t_cms"], context=ctx0)
    cm_t1, cm_t2, cm_t3 = gt.buffers["t_cms"].data

    # 生成 zeta
    tr.absorb_g1(cm_t1)
    tr.absorb_g1(cm_t2)
    tr.absorb_g1(cm_t3)
    zeta = tr.challenge(b"zeta")
    zeta_omega = (zeta * omega) % FR_MODULUS

    # 计算多项式在 zeta 点的值
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

    # 利用 v 折叠多项式
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

    # 生成 KZG 证明
    go = Graph()
    go.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=pk.srs)
    go.add_buffer(id="coeffs_list", device=Device.CPU, dtype=DType.OBJ, data=[list(combined_coeff), list(z_coeff)])
    go.add_buffer(id="z_list", device=Device.CPU, dtype=DType.OBJ, data=[int(zeta), int(zeta_omega)])
    go.add_node(op=OpType.KZG_OPEN_PREP_BATCH, inputs=["srs", "coeffs_list", "z_list"], outputs=["y_list", "q_list"], attrs=attrs0)
    go.add_node(op=OpType.KZG_BATCH_COMMIT, inputs=["srs", "q_list"], outputs=["pi_list"], attrs=attrs0)
    exe.run(go, pool=pool, trace=trace, keep=["y_list", "pi_list"], context=ctx0)
    y_check = int(go.buffers["y_list"].data[0]) % FR_MODULUS
    pi_zeta = go.buffers["pi_list"].data[0]
    if y_check % FR_MODULUS != combined_y % FR_MODULUS:
        raise ValueError("batch opening mismatch")

    pi_zeta_omega = go.buffers["pi_list"].data[1]

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

# 扩展变量求值函数。
# 遍历电路中的所有标准门（Gate），根据初始的 Witness 值和门类型（add, mul, scale, const 等），
# 推导出所有中间导线和输出导线的具体数值。
# 如果某些导线未被使用，则默认填充为 0，确保所有索引都有对应的域元素。
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

# 计算置换累积多项式 (Permutation Accumulator) Z 的求值序列。
# 利用挑战因子 beta 和 gamma，对标准索引（id1, id2, id3）和置换索引（sigma1, sigma2, sigma3）
# 下的导线值进行累乘。用于证明电路中不同门之间导线的连接一致性（Copy Constraints）。
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

# 计算核心商多项式 T(x) 并进行分段。
# 1. 为了避免除以零多项式 Z_H(x) 导致分母为 0，将所有多项式通过 Coset iNTT/NTT 转换到偏移大小为 4n 的陪集上。
# 2. 在计算图（Graph）中利用 PLONK_T_QUOTIENT_EVALS 算子，合并门约束和置换约束。
# 3. 将结果除以 Z_H 后转回系数表示。
# 4. 由于 T(x) 的度数最高可达 3n，需将其切割为 3 个度数小于 n 的多项式 t1, t2, t3 以适配 KZG 承诺容量。
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
    runtime_trace=None,
    runtime_pool=None,
    runtime_context=None,
    runtime_config: RuntimeConfig | None = None,
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

    l1_eval = [1] + [0] * (n - 1)
    l1_coeff = coeffs_from_evals_on_roots(l1_eval, omega=omega)

    reg = KernelRegistry()
    backend0 = runtime_config.backend if runtime_config is not None else Backend.CPU
    if runtime_context is not None:
        backend0 = runtime_context.backend
    register_cpu_kernels(reg, backend=backend0)
    if backend0 == Backend.METAL:
        from pyZKP.runtime.kernels.metal import register_metal_kernels
        register_metal_kernels(reg)
    exe = Executor(registry=reg)
    g = Graph()


    g.add_buffer(id="a_coeff", device=Device.CPU, dtype=DType.FR, data=list(a_coeff))
    g.add_buffer(id="b_coeff", device=Device.CPU, dtype=DType.FR, data=list(b_coeff))
    g.add_buffer(id="c_coeff", device=Device.CPU, dtype=DType.FR, data=list(c_coeff))
    g.add_buffer(id="z_coeff", device=Device.CPU, dtype=DType.FR, data=list(z_coeff))
    g.add_buffer(id="pi_coeff", device=Device.CPU, dtype=DType.FR, data=list(pi_coeff))

    g.add_buffer(id="ql_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_ql))
    g.add_buffer(id="qr_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_qr))
    g.add_buffer(id="qm_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_qm))
    g.add_buffer(id="qo_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_qo))
    g.add_buffer(id="qc_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_qc))

    g.add_buffer(id="s1_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_sigma1))
    g.add_buffer(id="s2_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_sigma2))
    g.add_buffer(id="s3_coeff", device=Device.CPU, dtype=DType.FR, data=list(pk.coeff_sigma3))

    g.add_buffer(id="l1_coeff", device=Device.CPU, dtype=DType.FR, data=list(l1_coeff))

    coset_attrs = {"n": m, "omega": omega_m, "shift": shift}
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["a_coeff"], outputs=["a_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["b_coeff"], outputs=["b_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["c_coeff"], outputs=["c_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["z_coeff"], outputs=["z_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["pi_coeff"], outputs=["pi_ext"], attrs=coset_attrs)

    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["ql_coeff"], outputs=["ql_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["qr_coeff"], outputs=["qr_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["qm_coeff"], outputs=["qm_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["qo_coeff"], outputs=["qo_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["qc_coeff"], outputs=["qc_ext"], attrs=coset_attrs)

    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["s1_coeff"], outputs=["s1_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["s2_coeff"], outputs=["s2_ext"], attrs=coset_attrs)
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["s3_coeff"], outputs=["s3_ext"], attrs=coset_attrs)

    shift_zw = (int(shift) * int(omega)) % FR_MODULUS
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["z_coeff"], outputs=["zshift_ext"], attrs={"n": m, "omega": omega_m, "shift": shift_zw})
    g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["l1_coeff"], outputs=["l1_ext"], attrs=coset_attrs)

    q_attrs = {
        "n": n,
        "m": m,
        "omega_m": omega_m,
        "shift": shift,
        "shift_n": pow(int(shift) % FR_MODULUS, n, FR_MODULUS),
        "alpha": int(alpha) % FR_MODULUS,
        "beta": int(beta) % FR_MODULUS,
        "gamma": int(gamma) % FR_MODULUS,
        "k1": int(c.k1) % FR_MODULUS,
        "k2": int(c.k2) % FR_MODULUS,
    }
    g.add_node(
        op=OpType.PLONK_T_QUOTIENT_EVALS,
        inputs=[
            "a_ext",
            "b_ext",
            "c_ext",
            "z_ext",
            "pi_ext",
            "ql_ext",
            "qr_ext",
            "qm_ext",
            "qo_ext",
            "qc_ext",
            "s1_ext",
            "s2_ext",
            "s3_ext",
            "zshift_ext",
            "l1_ext",
        ],
        outputs=["num_ext", "zh_ext"],
        attrs=q_attrs,
    )
    g.add_node(op=OpType.BATCH_INV, inputs=["zh_ext"], outputs=["inv_zh_ext"])
    g.add_node(op=OpType.POINTWISE_MUL, inputs=["num_ext", "inv_zh_ext"], outputs=["t_ext"])
    g.add_node(op=OpType.COSET_COEFFS_FROM_EVALS, inputs=["t_ext"], outputs=["t_coeff_full_m"], attrs={"omega": omega_m, "shift": shift})
    g.add_node(op=OpType.FROM_DEVICE, inputs=["t_coeff_full_m"], outputs=["t_coeff_full"])

    ctx0 = runtime_config.make_context(pool=runtime_pool, context=runtime_context) if runtime_config is not None else runtime_context
    exe.run(g, pool=runtime_pool, trace=runtime_trace, keep=["t_coeff_full"], context=ctx0)
    t_coeff_full = g.buffers["t_coeff_full"].data
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

# 批量多项式折叠函数（Batching / Folding）。
# 为了优化验证者的工作量，利用 Transcript 提供的随机挑战因子 v，
# 按 v 的幂次 (1, v, v^2...) 对所有的多项式系数、KZG 承诺（G1点）以及在 zeta 处的求值结果进行随机线性组合。
# 使得验证者只需进行一次 KZG 批量打开证明（Batch Opening Proof）校验。
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
