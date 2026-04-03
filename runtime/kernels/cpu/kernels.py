from __future__ import annotations

"""
CPU kernels：将项目内已有的 CPU reference/优化实现封装为 runtime 的可执行算子。

约定：
- 输入输出通过 Buffer 传递，算子参数通过 ctx["attrs"] 传递
- 若 ctx["pool"] 提供 CPUMemoryPool，则尽量复用 FR 的 list[int] 作为输出缓冲
"""

from typing import Any, Dict, List, Tuple

from common.crypto.field import FR_MODULUS, fr_batch_inv
from common.crypto.field.fr import fr_inv
from common.crypto.ecc.bn254 import G1_ZERO
from common.crypto.kzg.cpu_ref import SRS
from common.crypto.msm import (
    fixed_base_get_cached,
    fixed_base_precompute,
    msm_fixed_base,
    msm_fixed_base_batch,
    msm_naive_g1,
    msm_naive_g2,
    msm_pippenger,
    msm_pippenger_batch,
    msm_pippenger_g2,
)
from common.crypto.poly import (
    coeffs_from_evals_on_coset,
    coeffs_from_evals_on_roots,
    evals_from_coeffs_on_coset,
    evals_from_coeffs_on_roots,
    intt_inplace,
    ntt_inplace,
    poly_div_by_xn_minus_1,
    poly_eval,
    poly_mul_ntt,
)
from common.crypto.poly.cpu_ref import poly_sub
from runtime.ir.ops import OpType
from runtime.ir.types import Backend, Buffer, Device, DType
from runtime.memory import CPUMemoryPool
from runtime.kernels.registry import KernelRegistry
from runtime.metal import MetalBuffer

# 蒙哥马利表示法
_FR_P = int(FR_MODULUS)
_FR_MASK64 = (1 << 64) - 1
_FR_R = pow(2, 256, _FR_P)
_FR_RINV = pow(_FR_R, -1, _FR_P)


# 消除无意义的重复切片复制
# 全局缓存
_SRS_G1_PREFIX_CACHE: Dict[Tuple[int, int], Tuple[object, Tuple[Any, ...]]] = {}

# 缓存读取函数，接收原始SRS和需要的长度
def _srs_g1_prefix(s: SRS, n: int):
    k = (id(s.g1_powers), int(n))
    cached = _SRS_G1_PREFIX_CACHE.get(k)
    if cached is not None:
        ref, out = cached
        if ref is s.g1_powers:
            return out
    out = s.g1_powers[: int(n)]
    _SRS_G1_PREFIX_CACHE[k] = (s.g1_powers, out)
    return out


# cpu算子注册入口
def register_cpu_kernels(registry: KernelRegistry, *, backend: Backend = Backend.CPU) -> None:
    """
    注册所有 CPU 侧算子实现。可控制 backend。
    """
    registry.register(OpType.TO_DEVICE, Device.CPU, _to_device, backend=backend) 
    registry.register(OpType.FROM_DEVICE, Device.CPU, _from_device, backend=backend)
    registry.register(OpType.FROM_DEVICE, Device.METAL, _from_device, backend=backend)
    registry.register(OpType.ROOTS_EVALS_FROM_COEFFS, Device.CPU, _roots_evals_from_coeffs, backend=backend)
    registry.register(OpType.ROOTS_COEFFS_FROM_EVALS, Device.CPU, _roots_coeffs_from_evals, backend=backend)
    registry.register(OpType.COSET_EVALS_FROM_COEFFS, Device.CPU, _coset_evals_from_coeffs, backend=backend)
    registry.register(OpType.COSET_COEFFS_FROM_EVALS, Device.CPU, _coset_coeffs_from_evals, backend=backend)
    registry.register(OpType.BATCH_INV, Device.CPU, _batch_inv, backend=backend)
    registry.register(OpType.POINTWISE_MUL, Device.CPU, _pointwise_mul, backend=backend)
    registry.register(OpType.POLY_MUL_NTT, Device.CPU, _poly_mul_ntt, backend=backend)
    registry.register(OpType.POLY_SUB, Device.CPU, _poly_sub, backend=backend)
    registry.register(OpType.DIV_XN_MINUS_1, Device.CPU, _div_xn_minus_1, backend=backend)
    registry.register(OpType.PLONK_T_QUOTIENT_EVALS, Device.CPU, _plonk_t_quotient_evals, backend=backend)
    registry.register(OpType.MSM_G1, Device.CPU, _msm_g1, backend=backend)
    registry.register(OpType.MSM_G2, Device.CPU, _msm_g2, backend=backend)
    registry.register(OpType.MSM_G1_BATCH, Device.CPU, _msm_g1_batch, backend=backend)
    registry.register(OpType.KZG_COMMIT, Device.CPU, _kzg_commit, backend=backend)
    registry.register(OpType.KZG_OPEN, Device.CPU, _kzg_open, backend=backend)
    registry.register(OpType.KZG_OPEN_PREP_BATCH, Device.CPU, _kzg_open_prep_batch, backend=backend)
    registry.register(OpType.KZG_BATCH_COMMIT, Device.CPU, _kzg_batch_commit, backend=backend)
    registry.register(OpType.KZG_BATCH_OPEN, Device.CPU, _kzg_batch_open, backend=backend)


def _to_device(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    # 目前 to_device 主要处理 FR (标量)
    if inp.dtype != DType.FR:
        # 如果是 OBJ/G1/G2 且是 MSM 所需的点，我们暂时允许它透传而不做搬运，由 MSM 内部去 pack 
        if inp.dtype in (DType.OBJ, DType.G1, DType.G2):
            return {"outputs": {out_id: Buffer(id=out_id, device=Device.METAL, dtype=inp.dtype, data=inp.data)}}
        raise ValueError(f"to_device supports FR only, got {inp.dtype}")
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("to_device requires MetalContext with metal runtime")
    rt = c.metal

    host = inp.data
    n = len(host)
    out_len = n * 32
    
    pool = ctx.get("pool")
    if pool is not None and hasattr(pool, "alloc_metal"):
        mtl = pool.alloc_metal(rt, out_len)
    else:
        # 0 = MTLResourceStorageModeShared, 统一内存架构下 CPU/GPU 零拷贝共享
        mtl = rt.device.newBufferWithLength_options_(out_len, 0)
        
    if mtl is None:
        raise RuntimeError("failed to create MTLBuffer")
    
    ptr = mtl.contents()
    if ptr is None:
        raise RuntimeError("failed to map MTLBuffer contents")
    
    # 零拷贝写入：通过 memoryview 绕过中转数组，直接把 Montgomery 大数写入 Metal 的显存地址
    mv = ptr.as_buffer(out_len)
    buf = mv.cast("Q")
    
    for i in range(n):
        xm = (int(host[i]) * _FR_R) % _FR_P
        buf[i*4] = xm & _FR_MASK64
        buf[i*4+1] = (xm >> 64) & _FR_MASK64
        buf[i*4+2] = (xm >> 128) & _FR_MASK64
        buf[i*4+3] = (xm >> 192) & _FR_MASK64

    mb = MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=mtl)
    out = Buffer(id=out_id, device=Device.METAL, dtype=inp.dtype, data=mb, meta={"n": n})
    return {"outputs": {out_id: out}}


def _from_device(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    if inp.device == Device.CPU:
        return {"outputs": {out_id: Buffer(id=out_id, device=Device.CPU, dtype=inp.dtype, data=inp.data)}}
    if inp.dtype != DType.FR:
        if inp.dtype in (DType.OBJ, DType.G1, DType.G2):
            return {"outputs": {out_id: Buffer(id=out_id, device=Device.CPU, dtype=inp.dtype, data=inp.data)}}
        raise ValueError("from_device supports FR only")
    if not isinstance(inp.data, MetalBuffer):
        raise ValueError("from_device expects MetalBuffer")
    mtl = inp.data.mtl_buffer
    n = int(inp.data.n)
    ptr = mtl.contents()
    if ptr is None:
        raise RuntimeError("failed to map MTLBuffer contents")
    mv = ptr.as_buffer(n * 32)
    buf = mv.cast("Q")
    out_list: list[int] = []
    # 逐元素拼回蒙哥马利值并且还原为 FR 域元素
    for i in range(n):
        v = int(buf[i * 4]) | (int(buf[i * 4 + 1]) << 64) | (int(buf[i * 4 + 2]) << 128) | (int(buf[i * 4 + 3]) << 192)
        out_list.append(int((v * _FR_RINV) % _FR_P))
    out = Buffer(id=out_id, device=Device.CPU, dtype=inp.dtype, data=out_list, meta={"n": n})
    return {"outputs": {out_id: out}}


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
    acc = _msm_g1_impl(points.data, scalars.data, ctx["attrs"])
    return {"outputs": {out_id: Buffer(id=out_id, device=points.device, dtype=DType.G1, data=acc)}}


def _msm_g2(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    G2 MSM（多标量乘）。
    默认按规模阈值选择 naive 或 pippenger。
    """
    node = ctx["node"]
    points: Buffer = ctx["inputs"][0]
    scalars: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    acc = _msm_g2_impl(points.data, scalars.data, ctx["attrs"])
    return {"outputs": {out_id: Buffer(id=out_id, device=points.device, dtype=DType.G2, data=acc)}}


def _kzg_commit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    KZG 承诺：commit(srs, coeffs) -> G1。
    """
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    coeffs: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    s: SRS = srs.data
    scalars = [int(c) % FR_MODULUS for c in coeffs.data]
    if len(scalars) == 0:
        cm = G1_ZERO
    else:
        if len(scalars) > len(s.g1_powers):
            raise ValueError("SRS too small for polynomial degree")
        cm = _msm_g1_impl(_srs_g1_prefix(s, len(scalars)), scalars, ctx["attrs"])
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
    s: SRS = srs.data
    zz = int(z) % FR_MODULUS
    f = [int(c) % FR_MODULUS for c in coeffs.data]
    y = poly_eval(f, zz) % FR_MODULUS
    if len(f) == 0:
        pi = G1_ZERO
    else:
        f0 = list(f)
        f0[0] = (f0[0] - y) % FR_MODULUS
        q = _synthetic_division(f0, zz)
        if len(q) == 0:
            pi = G1_ZERO
        else:
            if len(q) > len(s.g1_powers):
                raise ValueError("SRS too small for quotient degree")
            pi = _msm_g1_impl(_srs_g1_prefix(s, len(q)), q, ctx["attrs"])
    return {
        "outputs": {
            y_id: Buffer(id=y_id, device=coeffs.device, dtype=DType.FR, data=int(y) % FR_MODULUS),
            pi_id: Buffer(id=pi_id, device=coeffs.device, dtype=DType.G1, data=pi),
        }
    }

def _kzg_open_prep_data(coeffs_list, z_list):
    if len(coeffs_list) != len(z_list):
        raise ValueError("length mismatch")
    ys = []
    qs = []
    for coeffs, z in zip(coeffs_list, z_list):
        zz = int(z) % FR_MODULUS
        f = [int(c) % FR_MODULUS for c in coeffs]
        y = poly_eval(f, zz) % FR_MODULUS
        if len(f) == 0:
            q = []
        else:
            f0 = list(f)
            f0[0] = (f0[0] - y) % FR_MODULUS
            q = _synthetic_division(f0, zz)
        ys.append(int(y) % FR_MODULUS)
        qs.append(q)
    return ys, qs


def _kzg_open_prep_batch(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    coeffs_list: Buffer = ctx["inputs"][1]
    z_list: Buffer = ctx["inputs"][2]
    y_id = node.outputs[0]
    q_id = node.outputs[1]

    ys, qs = _kzg_open_prep_data(coeffs_list.data, z_list.data)

    srs: Buffer = ctx["inputs"][0]
    return {
        "outputs": {
            y_id: Buffer(id=y_id, device=srs.device, dtype=DType.OBJ, data=ys),
            q_id: Buffer(id=q_id, device=srs.device, dtype=DType.OBJ, data=qs),
        }
    }


def _msm_g1_batch(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    points: Buffer = ctx["inputs"][0]
    scalars_list: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    outs = []
    for scalars in scalars_list.data:
        s = [int(c) % FR_MODULUS for c in scalars]
        outs.append(_msm_g1_impl(points.data[: len(s)], s, ctx["attrs"]) if len(s) != 0 else G1_ZERO)
    return {"outputs": {out_id: Buffer(id=out_id, device=points.device, dtype=DType.OBJ, data=outs)}}


def _kzg_batch_commit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    polys: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    s: SRS = srs.data
    scalars_list = []
    max_len = 0
    for coeffs in polys.data:
        scalars = [int(c) % FR_MODULUS for c in coeffs]
        scalars_list.append(scalars)
        if len(scalars) > max_len:
            max_len = len(scalars)
    if max_len > len(s.g1_powers):
        raise ValueError("SRS too small for polynomial degree")

    threshold = int(ctx["attrs"].get("pippenger_threshold", 64))
    if len(scalars_list) >= 2 and max_len >= threshold:
        points = _srs_g1_prefix(s, max_len)
        fb = ctx["attrs"].get("fixed_base")
        if fb is True:
            fb_w = int(ctx["attrs"].get("fixed_base_window_bits", 8))
            pre = fixed_base_precompute(points, fb_w)
            outs = msm_fixed_base_batch(pre, scalars_list)
        elif fb is None and isinstance(points, tuple):
            fb_w = int(ctx["attrs"].get("fixed_base_window_bits", 8))
            pre = fixed_base_get_cached(points, fb_w)
            if pre is not None:
                outs = msm_fixed_base_batch(pre, scalars_list)
            else:
                if "window_bits" in ctx["attrs"]:
                    window_bits = int(ctx["attrs"]["window_bits"])
                else:
                    window_bits = max(4, min(16, int(max_len).bit_length() - 2))
                outs = msm_pippenger_batch(points, scalars_list, window_bits=window_bits)
        else:
            if "window_bits" in ctx["attrs"]:
                window_bits = int(ctx["attrs"]["window_bits"])
            else:
                window_bits = max(4, min(16, int(max_len).bit_length() - 2))
            outs = msm_pippenger_batch(points, scalars_list, window_bits=window_bits)
    else:
        outs = [_msm_g1_impl(_srs_g1_prefix(s, len(sc)), sc, ctx["attrs"]) if len(sc) != 0 else G1_ZERO for sc in scalars_list]
    return {"outputs": {out_id: Buffer(id=out_id, device=srs.device, dtype=DType.OBJ, data=outs)}}


def _kzg_batch_open(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    coeffs_list: Buffer = ctx["inputs"][1]
    z_list: Buffer = ctx["inputs"][2]
    y_id = node.outputs[0]
    pi_id = node.outputs[1]

    s: SRS = srs.data
    ys, qs = _kzg_open_prep_data(coeffs_list.data, z_list.data)

    max_len = 0
    for q in qs:
        if len(q) > max_len:
            max_len = len(q)
    if max_len > len(s.g1_powers):
        raise ValueError("SRS too small for quotient degree")

    threshold = int(ctx["attrs"].get("pippenger_threshold", 64))
    if len(qs) >= 2 and max_len >= threshold:
        points = _srs_g1_prefix(s, max_len)
        fb = ctx["attrs"].get("fixed_base")
        if fb is True:
            fb_w = int(ctx["attrs"].get("fixed_base_window_bits", 8))
            pre = fixed_base_precompute(points, fb_w)
            pis = msm_fixed_base_batch(pre, qs)
        elif fb is None and isinstance(points, tuple):
            fb_w = int(ctx["attrs"].get("fixed_base_window_bits", 8))
            pre = fixed_base_get_cached(points, fb_w)
            if pre is not None:
                pis = msm_fixed_base_batch(pre, qs)
            else:
                if "window_bits" in ctx["attrs"]:
                    window_bits = int(ctx["attrs"]["window_bits"])
                else:
                    window_bits = max(4, min(16, int(max_len).bit_length() - 2))
                pis = msm_pippenger_batch(points, qs, window_bits=window_bits)
        else:
            if "window_bits" in ctx["attrs"]:
                window_bits = int(ctx["attrs"]["window_bits"])
            else:
                window_bits = max(4, min(16, int(max_len).bit_length() - 2))
            pis = msm_pippenger_batch(points, qs, window_bits=window_bits)
    else:
        pis = [_msm_g1_impl(_srs_g1_prefix(s, len(q)), q, ctx["attrs"]) if len(q) != 0 else G1_ZERO for q in qs]

    return {
        "outputs": {
            y_id: Buffer(id=y_id, device=srs.device, dtype=DType.OBJ, data=ys),
            pi_id: Buffer(id=pi_id, device=srs.device, dtype=DType.OBJ, data=pis),
        }
    }


def _msm_g1_impl(points, scalars, attrs: Dict[str, Any]):
    threshold = int(attrs.get("pippenger_threshold", 64))
    if len(points) >= threshold:
        if "window_bits" in attrs:
            window_bits = int(attrs["window_bits"])
        else:
            window_bits = max(4, min(16, int(len(points)).bit_length() - 2))
        fb = attrs.get("fixed_base")
        if fb is True and isinstance(points, tuple):
            fb_w = int(attrs.get("fixed_base_window_bits", min(8, window_bits)))
            pre = fixed_base_precompute(points, fb_w)
            return msm_fixed_base(pre, scalars)
        if fb is None and isinstance(points, tuple):
            fb_w = int(attrs.get("fixed_base_window_bits", min(8, window_bits)))
            pre = fixed_base_get_cached(points, fb_w)
            if pre is not None:
                return msm_fixed_base(pre, scalars)
        return msm_pippenger(points, scalars, window_bits=window_bits)
    return msm_naive_g1(points, scalars)


def _msm_g2_impl(points, scalars, attrs: Dict[str, Any]):
    threshold = int(attrs.get("pippenger_threshold", 64))
    if len(points) >= threshold:
        if "window_bits" in attrs:
            window_bits = int(attrs["window_bits"])
        else:
            window_bits = max(4, min(16, int(len(points)).bit_length() - 2))
        return msm_pippenger_g2(points, scalars, window_bits=window_bits)
    return msm_naive_g2(points, scalars)


def _synthetic_division(coeffs: List[int], z: int) -> List[int]:
    if len(coeffs) <= 1:
        return []
    z = int(z) % FR_MODULUS
    out = [0] * (len(coeffs) - 1)
    acc = coeffs[-1] % FR_MODULUS
    out[-1] = acc
    for i in range(len(coeffs) - 2, 0, -1):
        acc = (coeffs[i] + acc * z) % FR_MODULUS
        out[i - 1] = acc
    return out


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
