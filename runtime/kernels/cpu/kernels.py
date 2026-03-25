from __future__ import annotations

"""
CPU kernels：将项目内已有的 CPU reference/优化实现封装为 runtime 的可执行算子。

约定：
- 输入输出通过 Buffer 传递，算子参数通过 ctx["attrs"] 传递
- 若 ctx["pool"] 提供 CPUMemoryPool，则尽量复用 FR 的 list[int] 作为输出缓冲
"""

from typing import Any, Dict, List

from pyZKP.common.crypto.field import FR_MODULUS, fr_batch_inv
from pyZKP.common.crypto.field.fr import fr_inv
from pyZKP.common.crypto.kzg.cpu_ref import commit as kzg_commit, open_proof as kzg_open
from pyZKP.common.crypto.msm import msm_naive_g1, msm_naive_g2, msm_pippenger
from pyZKP.common.crypto.poly import (
    coeffs_from_evals_on_coset,
    coeffs_from_evals_on_roots,
    evals_from_coeffs_on_coset,
    evals_from_coeffs_on_roots,
    intt_inplace,
    ntt_inplace,
    poly_div_by_xn_minus_1,
    poly_mul_ntt,
)
from pyZKP.common.crypto.poly.cpu_ref import poly_sub
from pyZKP.runtime.ir.ops import OpType
from pyZKP.runtime.ir.types import Buffer, Device, DType
from pyZKP.runtime.memory import CPUMemoryPool
from pyZKP.runtime.kernels.registry import KernelRegistry


# cpu算子注册入口
def register_cpu_kernels(registry: KernelRegistry) -> None:
    """
    注册所有 CPU 侧算子实现。
    """
    registry.register(OpType.ROOTS_EVALS_FROM_COEFFS, Device.CPU, _roots_evals_from_coeffs)
    registry.register(OpType.ROOTS_COEFFS_FROM_EVALS, Device.CPU, _roots_coeffs_from_evals)
    registry.register(OpType.COSET_EVALS_FROM_COEFFS, Device.CPU, _coset_evals_from_coeffs)
    registry.register(OpType.COSET_COEFFS_FROM_EVALS, Device.CPU, _coset_coeffs_from_evals)
    registry.register(OpType.BATCH_INV, Device.CPU, _batch_inv)
    registry.register(OpType.POINTWISE_MUL, Device.CPU, _pointwise_mul)
    registry.register(OpType.POLY_MUL_NTT, Device.CPU, _poly_mul_ntt)
    registry.register(OpType.POLY_SUB, Device.CPU, _poly_sub)
    registry.register(OpType.DIV_XN_MINUS_1, Device.CPU, _div_xn_minus_1)
    registry.register(OpType.PLONK_T_QUOTIENT_EVALS, Device.CPU, _plonk_t_quotient_evals)
    registry.register(OpType.MSM_G1, Device.CPU, _msm_g1)
    registry.register(OpType.MSM_G2, Device.CPU, _msm_g2)
    registry.register(OpType.KZG_COMMIT, Device.CPU, _kzg_commit)
    registry.register(OpType.KZG_OPEN, Device.CPU, _kzg_open)


def _roots_evals_from_coeffs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    在单位根域 H 上做 NTT：coeffs -> evals。
    """
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    n = int(ctx["attrs"]["n"])
    omega = int(ctx["attrs"]["omega"])
    out_id = node.outputs[0]
    pool: CPUMemoryPool | None = ctx.get("pool")
    ev = _alloc_fr(pool, n)
    for i in range(min(n, len(inp.data))):
        ev[i] = int(inp.data[i]) % FR_MODULUS
    ntt_inplace(ev, omega)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=ev, meta={"n": n, "omega": omega})}}


def _roots_coeffs_from_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    在单位根域 H 上做 INTT：evals -> coeffs。
    """
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    omega = int(ctx["attrs"]["omega"])
    out_id = node.outputs[0]
    pool: CPUMemoryPool | None = ctx.get("pool")
    coeff = _alloc_fr(pool, len(inp.data))
    for i in range(len(inp.data)):
        coeff[i] = int(inp.data[i]) % FR_MODULUS
    intt_inplace(coeff, omega)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=coeff, meta={"omega": omega})}}


def _coset_evals_from_coeffs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    在陪集上求值：evals[i] = f(shift * omega^i)。
    实现方式：先对系数按 shift^i 缩放，再做普通 NTT。
    """
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    n = int(ctx["attrs"]["n"])
    omega = int(ctx["attrs"]["omega"])
    shift = int(ctx["attrs"]["shift"])
    out_id = node.outputs[0]
    pool: CPUMemoryPool | None = ctx.get("pool")
    ev = _alloc_fr(pool, n)
    ss = int(shift) % FR_MODULUS
    pow_s = 1
    for i in range(min(n, len(inp.data))):
        ev[i] = (int(inp.data[i]) % FR_MODULUS) * pow_s % FR_MODULUS
        pow_s = (pow_s * ss) % FR_MODULUS
    ntt_inplace(ev, omega)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=ev, meta={"n": n, "omega": omega, "shift": shift})}}


def _coset_coeffs_from_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    在陪集上插值：由 evals[i] = f(shift * omega^i) 还原系数。
    """
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    omega = int(ctx["attrs"]["omega"])
    shift = int(ctx["attrs"]["shift"])
    out_id = node.outputs[0]
    pool: CPUMemoryPool | None = ctx.get("pool")
    coeff = _alloc_fr(pool, len(inp.data))
    for i in range(len(inp.data)):
        coeff[i] = int(inp.data[i]) % FR_MODULUS
    intt_inplace(coeff, omega)
    ss = int(shift) % FR_MODULUS
    inv_s = fr_inv(ss) if ss != 0 else 0
    pow_inv_s = 1
    for i in range(len(coeff)):
        coeff[i] = coeff[i] * pow_inv_s % FR_MODULUS
        pow_inv_s = (pow_inv_s * inv_s) % FR_MODULUS if inv_s != 0 else 0
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=coeff, meta={"omega": omega, "shift": shift})}}


def _batch_inv(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    批量求逆（Montgomery trick）。
    输入为 0 的位置输出 0。
    """
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    pool: CPUMemoryPool | None = ctx.get("pool")
    invs = fr_batch_inv(inp.data)
    if pool is not None:
        out = _alloc_fr(pool, len(invs))
        for i in range(len(invs)):
            out[i] = invs[i]
        return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=out, meta=inp.meta)}}
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=invs, meta=inp.meta)}}


def _pointwise_mul(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    点值域逐点乘：out[i] = a[i] * b[i]。
    """
    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    if len(a.data) != len(b.data):
        raise ValueError("length mismatch")
    pool: CPUMemoryPool | None = ctx.get("pool")
    out = _alloc_fr(pool, len(a.data))
    for i in range(len(a.data)):
        out[i] = (int(a.data[i]) * int(b.data[i])) % FR_MODULUS
    return {"outputs": {out_id: Buffer(id=out_id, device=a.device, dtype=DType.FR, data=out)}}


def _poly_mul_ntt(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    多项式卷积乘法（系数域），内部使用 NTT。
    """
    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    out = poly_mul_ntt(a.data, b.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=a.device, dtype=DType.FR, data=out)}}


def _poly_sub(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    多项式减法（系数域）。
    """
    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    out = poly_sub(a.data, b.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=a.device, dtype=DType.FR, data=out)}}


def _div_xn_minus_1(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    专用除法：num / (X^n - 1)，返回商 q 与余数 r。
    """
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    n = int(ctx["attrs"]["n"])
    q_id = node.outputs[0]
    r_id = node.outputs[1]
    q, r = poly_div_by_xn_minus_1(inp.data, n)
    return {"outputs": {q_id: Buffer(id=q_id, device=inp.device, dtype=DType.FR, data=q), r_id: Buffer(id=r_id, device=inp.device, dtype=DType.FR, data=r)}}


def _msm_g1(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    G1 MSM（多标量乘）。
    默认按规模阈值选择 naive 或 pippenger。
    """
    node = ctx["node"]
    points: Buffer = ctx["inputs"][0]
    scalars: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    threshold = int(ctx["attrs"].get("pippenger_threshold", 1024))
    if len(points.data) >= threshold:
        window_bits = int(ctx["attrs"].get("window_bits", 16))
        acc = msm_pippenger(points.data, scalars.data, window_bits=window_bits)
    else:
        acc = msm_naive_g1(points.data, scalars.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=points.device, dtype=DType.G1, data=acc)}}


def _msm_g2(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    G2 MSM（多标量乘）。
    当前仍用 naive 作为基线实现。
    """
    node = ctx["node"]
    points: Buffer = ctx["inputs"][0]
    scalars: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    acc = msm_naive_g2(points.data, scalars.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=points.device, dtype=DType.G2, data=acc)}}


def _kzg_commit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    KZG 承诺：commit(srs, coeffs) -> G1。
    """
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    coeffs: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    cm = kzg_commit(srs.data, coeffs.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=coeffs.device, dtype=DType.G1, data=cm)}}


def _kzg_open(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    KZG 单点打开证明：open(srs, coeffs, z) -> (y, proof)。
    """
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    coeffs: Buffer = ctx["inputs"][1]
    z = int(ctx["attrs"]["z"])
    y_id = node.outputs[0]
    pi_id = node.outputs[1]
    y, pi = kzg_open(srs.data, coeffs.data, z)
    return {
        "outputs": {
            y_id: Buffer(id=y_id, device=coeffs.device, dtype=DType.FR, data=int(y) % FR_MODULUS),
            pi_id: Buffer(id=pi_id, device=coeffs.device, dtype=DType.G1, data=pi),
        }
    }


def _alloc_fr(pool: CPUMemoryPool | None, n: int) -> List[int]:
    """
    从内存池分配 FR 数组；若 pool 为空则直接新建 list。
    """
    if pool is None:
        return [0] * int(n)
    return pool.alloc_fr(int(n))


def _plonk_t_quotient_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    PLONK quotient 融合算子（点值域）：
    给定扩展域陪集上的各多项式点值，逐点构造 num(x) 与 zh(x)=x^n-1。
    """
    node = ctx["node"]
    attrs = ctx["attrs"]
    n = int(attrs["n"])
    m = int(attrs["m"])
    omega_m = int(attrs["omega_m"])
    shift = int(attrs["shift"])
    shift_n = int(attrs["shift_n"])
    alpha = int(attrs["alpha"]) % FR_MODULUS
    beta = int(attrs["beta"]) % FR_MODULUS
    gamma = int(attrs["gamma"]) % FR_MODULUS
    k1 = int(attrs["k1"]) % FR_MODULUS
    k2 = int(attrs["k2"]) % FR_MODULUS
    u = pow(int(omega_m) % FR_MODULUS, n, FR_MODULUS)

    (
        a_ext,
        b_ext,
        c_ext,
        z_ext,
        pi_ext,
        ql_ext,
        qr_ext,
        qm_ext,
        qo_ext,
        qc_ext,
        s1_ext,
        s2_ext,
        s3_ext,
        zshift_ext,
        l1_ext,
    ) = [ctx["inputs"][i].data for i in range(15)]

    if not all(len(v) == m for v in [a_ext, b_ext, c_ext, z_ext, pi_ext, ql_ext, qr_ext, qm_ext, qo_ext, qc_ext, s1_ext, s2_ext, s3_ext, zshift_ext, l1_ext]):
        raise ValueError("eval length mismatch")

    alpha2 = (alpha * alpha) % FR_MODULUS

    x = shift % FR_MODULUS
    u_pow = 1
    zh: List[int] = []
    num: List[int] = []
    for i in range(m):
        a = int(a_ext[i]) % FR_MODULUS
        b = int(b_ext[i]) % FR_MODULUS
        cc = int(c_ext[i]) % FR_MODULUS
        z = int(z_ext[i]) % FR_MODULUS

        gate = (int(ql_ext[i]) * a + int(qr_ext[i]) * b) % FR_MODULUS
        gate = (gate + int(qm_ext[i]) * a % FR_MODULUS * b) % FR_MODULUS
        gate = (gate + int(qo_ext[i]) * cc + int(qc_ext[i]) + int(pi_ext[i])) % FR_MODULUS

        t1 = (a + beta * x + gamma) % FR_MODULUS
        t2 = (b + beta * k1 % FR_MODULUS * x + gamma) % FR_MODULUS
        t3 = (cc + beta * k2 % FR_MODULUS * x + gamma) % FR_MODULUS
        left = (t1 * t2) % FR_MODULUS
        left = (left * t3) % FR_MODULUS
        left = (left * z) % FR_MODULUS

        u1v = (a + beta * int(s1_ext[i]) + gamma) % FR_MODULUS
        u2v = (b + beta * int(s2_ext[i]) + gamma) % FR_MODULUS
        u3v = (cc + beta * int(s3_ext[i]) + gamma) % FR_MODULUS
        right = (u1v * u2v) % FR_MODULUS
        right = (right * u3v) % FR_MODULUS
        right = (right * int(zshift_ext[i])) % FR_MODULUS

        perm = (left - right) % FR_MODULUS
        boundary = ((z - 1) % FR_MODULUS) * (int(l1_ext[i]) % FR_MODULUS) % FR_MODULUS
        num.append((gate + alpha * perm + alpha2 * boundary) % FR_MODULUS)

        x_n = (shift_n * u_pow) % FR_MODULUS
        zh.append((x_n - 1) % FR_MODULUS)

        u_pow = (u_pow * u) % FR_MODULUS
        x = (x * omega_m) % FR_MODULUS

    out_num = node.outputs[0]
    out_zh = node.outputs[1]
    return {
        "outputs": {
            out_num: Buffer(id=out_num, device=Device.CPU, dtype=DType.FR, data=num, meta={"m": m}),
            out_zh: Buffer(id=out_zh, device=Device.CPU, dtype=DType.FR, data=zh, meta={"m": m}),
        }
    }
