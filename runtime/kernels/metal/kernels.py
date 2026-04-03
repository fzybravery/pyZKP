from __future__ import annotations

from typing import Any, Dict

from runtime.ir.ops import OpType
from runtime.ir.types import Backend, Buffer, Device, DType
from runtime.kernels.registry import KernelRegistry
from runtime.metal import MetalBuffer
from crypto.field.fr import FR_MODULUS


def register_metal_kernels(registry: KernelRegistry) -> None:
    registry.register(OpType.POINTWISE_MUL, Device.METAL, _pointwise_mul, backend=Backend.METAL) # 256-bit 域元素乘法，Montgomery 表示法
    registry.register(OpType.POLY_SUB, Device.METAL, _poly_sub, backend=Backend.METAL)
    registry.register(OpType.ROOTS_EVALS_FROM_COEFFS, Device.METAL, _roots_evals_from_coeffs, backend=Backend.METAL) # 256-bit 域元素的根计算
    registry.register(OpType.ROOTS_COEFFS_FROM_EVALS, Device.METAL, _roots_coeffs_from_evals, backend=Backend.METAL) # 256-bit 域元素的系数计算
    registry.register(OpType.COSET_EVALS_FROM_COEFFS, Device.METAL, _coset_evals_from_coeffs, backend=Backend.METAL)
    registry.register(OpType.COSET_COEFFS_FROM_EVALS, Device.METAL, _coset_coeffs_from_evals, backend=Backend.METAL)
    registry.register(OpType.MSM_G1, Device.METAL, _msm_g1, backend=Backend.METAL)
    registry.register(OpType.KZG_BATCH_COMMIT, Device.METAL, _kzg_batch_commit, backend=Backend.METAL)

def _kzg_batch_commit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    from runtime.metal.msm import metal_msm_g1, metal_msm_g1_v2
    from runtime.kernels.cpu.kernels import _srs_g1_prefix
    from crypto.ecc.bn254 import G1_ZERO
    from crypto.field.fr import FR_MODULUS
    import array
    
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    polys: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal kzg_batch_commit requires MetalContext")
    rt = c.metal
    msm_mode = getattr(c.config, "metal_msm_mode", "v1") if c.config else "v1"
    
    s = srs.data
    scalars_list = []
    max_len = 0
    for coeffs in polys.data:
        scalars = [int(x) % FR_MODULUS for x in coeffs]
        scalars_list.append(scalars)
        if len(scalars) > max_len:
            max_len = len(scalars)
    if max_len > len(s.g1_powers):
        raise ValueError("SRS too small for polynomial degree")

    outs = []
    for sc in scalars_list:
        if len(sc) == 0:
            outs.append(G1_ZERO)
            continue
            
        points = _srs_g1_prefix(s, len(sc))
        
        # 转换 sc 到 Montgomery 并上传 Metal
        _FR_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        _FR_R = pow(2, 256, _FR_P)
        _FR_MASK64 = (1 << 64) - 1
        
        flat = []
        for x in sc:
            xm = (x * _FR_R) % _FR_P
            flat.append(int(xm & _FR_MASK64))
            flat.append(int((xm >> 64) & _FR_MASK64))
            flat.append(int((xm >> 128) & _FR_MASK64))
            flat.append(int((xm >> 192) & _FR_MASK64))
        arr = array.array("Q", flat)
        b = arr.tobytes()
        mtl_buffer = rt.device.newBufferWithBytes_length_options_(b, len(b), 0)
        
        # 调用 Metal MSM
        if msm_mode == "v2":
            acc = metal_msm_g1_v2(rt, points, mtl_buffer, len(sc))
        else:
            acc = metal_msm_g1(rt, points, mtl_buffer, len(sc))
        outs.append(acc)
        
    return {"outputs": {out_id: Buffer(id=out_id, device=Device.CPU, dtype=DType.OBJ, data=outs)}}

def _msm_g1(ctx: Dict[str, Any]) -> Dict[str, Any]:
    from runtime.metal.msm import metal_msm_g1, metal_msm_g1_v2
    
    node = ctx["node"]
    points: Buffer = ctx["inputs"][0]
    scalars: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal msm_g1 requires MetalContext")
    rt = c.metal
    
    msm_mode = getattr(c.config, "metal_msm_mode", "v1") if c.config else "v1"

    # 1. 检查数据类型
    if scalars.dtype != DType.FR:
        raise ValueError("metal msm_g1 scalars must be FR")
        
    # 2. 如果 scalars 已经在显存中 (MetalBuffer)，直接用；如果还在 CPU，这需要 TO_DEVICE
    # 在我们的自动图重写逻辑中，它应该已经是 MetalBuffer 了。
    if not isinstance(scalars.data, MetalBuffer):
        raise ValueError("metal msm_g1 expects scalars to be MetalBuffer")
        
    num_points = int(scalars.data.n)
    if num_points > len(points.data):
        raise ValueError("scalars length exceeds points length")

    # 3. 调用 Metal MSM
    if msm_mode == "v2":
        acc = metal_msm_g1_v2(rt, points.data[:num_points], scalars.data.mtl_buffer, num_points)
    else:
        acc = metal_msm_g1(rt, points.data[:num_points], scalars.data.mtl_buffer, num_points)
    
    # 4. G1 结果通常是一个单独的点，且后续操作 (如 KZG Verification) 通常在 CPU 上
    # 为了简化，我们暂时让它返回在 CPU 上的 G1 结果。
    return {"outputs": {out_id: Buffer(id=out_id, device=Device.CPU, dtype=DType.G1, data=acc)}}

def _poly_sub(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    if a.dtype != DType.FR or b.dtype != DType.FR:
        raise ValueError("metal poly_sub supports FR only")
    if not isinstance(a.data, MetalBuffer) or not isinstance(b.data, MetalBuffer):
        raise ValueError("metal poly_sub expects MetalBuffer")
    if len(a.data) != len(b.data):
        raise ValueError("length mismatch")
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal poly_sub requires MetalContext with metal runtime")
    rt = c.metal

    n = int(a.data.n)
    out_len = n * 32
    out_mtl = _alloc_metal(ctx, rt, out_len)
    if out_mtl is None:
        raise RuntimeError("failed to allocate output MTLBuffer")

    cmd = rt.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(rt.pso_poly_sub_fr_mont)
    enc.setBuffer_offset_atIndex_(a.data.mtl_buffer, 0, 0)
    enc.setBuffer_offset_atIndex_(b.data.mtl_buffer, 0, 1)
    enc.setBuffer_offset_atIndex_(out_mtl, 0, 2)
    nb = struct.pack("I", int(n))
    enc.setBytes_length_atIndex_(nb, len(nb), 3)

    w = int(rt.pso_poly_sub_fr_mont.threadExecutionWidth())
    if w <= 0:
        w = 64
    tg = Metal.MTLSizeMake(int(w), 1, 1)
    grid = Metal.MTLSizeMake(int((n + w - 1) // w * w), 1, 1)
    enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    out = Buffer(id=out_id, device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=out_mtl), meta={"n": n})
    return {"outputs": {out_id: out}}

def _alloc_metal(ctx: Dict[str, Any], rt: Any, size: int) -> Any:
    pool = ctx.get("pool")
    if pool is not None and hasattr(pool, "alloc_metal"):
        return pool.alloc_metal(rt, size)
    return rt.device.newBufferWithLength_options_(size, 0)

def _pointwise_mul(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    if a.dtype != DType.FR or b.dtype != DType.FR:
        raise ValueError("metal pointwise_mul supports FR only")
    if not isinstance(a.data, MetalBuffer) or not isinstance(b.data, MetalBuffer):
        raise ValueError("metal pointwise_mul expects MetalBuffer")
    if len(a.data) != len(b.data):
        raise ValueError("length mismatch")
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal pointwise_mul requires MetalContext with metal runtime")
    # 获取 MetalRuntime
    rt = c.metal

    n = int(a.data.n)
    out_len = n * 32
    out_mtl = _alloc_metal(ctx, rt, out_len)
    if out_mtl is None:
        raise RuntimeError("failed to allocate output MTLBuffer")

    cmd = rt.queue.commandBuffer()  # 命令缓冲区
    enc = cmd.computeCommandEncoder()  # 计算命令编码器
    enc.setComputePipelineState_(rt.pso_pointwise_mul_fr_mont)
    enc.setBuffer_offset_atIndex_(a.data.mtl_buffer, 0, 0)
    enc.setBuffer_offset_atIndex_(b.data.mtl_buffer, 0, 1)
    enc.setBuffer_offset_atIndex_(out_mtl, 0, 2)
    nb = struct.pack("I", int(n))
    enc.setBytes_length_atIndex_(nb, len(nb), 3)

    # 计算 dispatch 规模
    w = int(rt.pso_pointwise_mul_fr_mont.threadExecutionWidth())
    if w <= 0:
        w = 64
    tg = Metal.MTLSizeMake(int(w), 1, 1) # 每个线程组的线程数
    grid = Metal.MTLSizeMake(int((n + w - 1) // w * w), 1, 1) # 总线程数，对齐到 w
    enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    out = Buffer(id=out_id, device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=out_mtl), meta={"n": n})
    return {"outputs": {out_id: out}}


def _coset_evals_from_coeffs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import array
    import math
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    
    n = int(ctx["attrs"]["n"])
    in_size = int(inp.data.n) if isinstance(inp.data, MetalBuffer) else n
    
    omega = int(ctx["attrs"]["omega"])
    shift = int(ctx["attrs"]["shift"])
    if inp.dtype != DType.FR:
        raise ValueError("metal coset_evals_from_coeffs supports FR only")
    if not isinstance(inp.data, MetalBuffer):
        raise ValueError("metal coset_evals_from_coeffs expects MetalBuffer")
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal requires MetalContext with metal runtime")
    rt = c.metal

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    logn = int(math.log2(n))
    if (1 << logn) != n:
        raise ValueError("n must be a power of two")

    p = int(FR_MODULUS)
    r = pow(2, 256, p)

    # 1. 准备 shift 标量 (Montgomery 形式)
    shift_m = (shift * r) % p
    shift_bytes = array.array(
        "Q",
        [
            int(shift_m & ((1 << 64) - 1)),
            int((shift_m >> 64) & ((1 << 64) - 1)),
            int((shift_m >> 128) & ((1 << 64) - 1)),
            int((shift_m >> 192) & ((1 << 64) - 1)),
        ],
    ).tobytes()

    scaled_mtl = _alloc_metal(ctx, rt, n * 32)
    nb = struct.pack("I", int(n))
    in_size_b = struct.pack("I", int(in_size))

    cmd = rt.queue.commandBuffer()
    
    # 步骤 A: poly_scale_shift
    enc1 = cmd.computeCommandEncoder()
    enc1.setComputePipelineState_(rt.pso_poly_scale_shift_fr_mont)
    enc1.setBuffer_offset_atIndex_(inp.data.mtl_buffer, 0, 0)
    enc1.setBuffer_offset_atIndex_(scaled_mtl, 0, 1)
    enc1.setBytes_length_atIndex_(shift_bytes, len(shift_bytes), 2)
    enc1.setBytes_length_atIndex_(nb, len(nb), 3)
    enc1.setBytes_length_atIndex_(in_size_b, len(in_size_b), 4)
    w = int(rt.pso_poly_scale_shift_fr_mont.threadExecutionWidth())
    if w <= 0: w = 64
    tg1 = Metal.MTLSizeMake(w, 1, 1)
    grid1 = Metal.MTLSizeMake((n + w - 1) // w * w, 1, 1)
    enc1.dispatchThreads_threadsPerThreadgroup_(grid1, tg1)
    enc1.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    # 步骤 B: roots_evals_from_coeffs (NTT)
    # 创建一个包含 scaled_mtl 的临时 Buffer
    temp_buf = Buffer(id="temp", device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=scaled_mtl))
    temp_ctx = dict(ctx)
    temp_ctx["inputs"] = [temp_buf]
    
    # 调用现有的 roots_evals_from_coeffs
    res = _roots_evals_from_coeffs(temp_ctx)
    out_buf = res["outputs"][out_id]
    out_buf.meta["shift"] = shift
    return {"outputs": {out_id: out_buf}}


def _coset_coeffs_from_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import array
    import math
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    
    n = int(inp.data.n) if isinstance(inp.data, MetalBuffer) else int(ctx["attrs"].get("n", 0))
    in_size = n
    
    omega = int(ctx["attrs"]["omega"])
    shift = int(ctx["attrs"]["shift"])
    if inp.dtype != DType.FR:
        raise ValueError("metal coset_coeffs_from_evals supports FR only")
    if not isinstance(inp.data, MetalBuffer):
        raise ValueError("metal coset_coeffs_from_evals expects MetalBuffer")
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal requires MetalContext")
    rt = c.metal

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    logn = int(math.log2(n))
    if (1 << logn) != n:
        raise ValueError("n must be a power of two")

    p = int(FR_MODULUS)
    r = pow(2, 256, p)

    inv_shift = pow(int(shift) % p, -1, p) if (int(shift) % p) != 0 else 0
    inv_shift_m = (inv_shift * r) % p
    inv_shift_bytes = array.array(
        "Q",
        [
            int(inv_shift_m & ((1 << 64) - 1)),
            int((inv_shift_m >> 64) & ((1 << 64) - 1)),
            int((inv_shift_m >> 128) & ((1 << 64) - 1)),
            int((inv_shift_m >> 192) & ((1 << 64) - 1)),
        ],
    ).tobytes()

    # 步骤 A: roots_coeffs_from_evals (INTT)
    res = _roots_coeffs_from_evals(ctx)
    intt_buf = res["outputs"][out_id]
    intt_mtl = intt_buf.data.mtl_buffer

    out_mtl = _alloc_metal(ctx, rt, n * 32)
    nb = struct.pack("I", int(n))
    in_size_b = struct.pack("I", int(n)) # intt output is size n

    cmd = rt.queue.commandBuffer()

    # 步骤 B: poly_scale_shift (使用 inv_shift)
    enc2 = cmd.computeCommandEncoder()
    enc2.setComputePipelineState_(rt.pso_poly_scale_shift_fr_mont)
    enc2.setBuffer_offset_atIndex_(intt_mtl, 0, 0)
    enc2.setBuffer_offset_atIndex_(out_mtl, 0, 1)
    enc2.setBytes_length_atIndex_(inv_shift_bytes, len(inv_shift_bytes), 2)
    enc2.setBytes_length_atIndex_(nb, len(nb), 3)
    enc2.setBytes_length_atIndex_(in_size_b, len(in_size_b), 4)
    w = int(rt.pso_poly_scale_shift_fr_mont.threadExecutionWidth())
    if w <= 0: w = 64
    tg2 = Metal.MTLSizeMake(w, 1, 1)
    grid2 = Metal.MTLSizeMake((n + w - 1) // w * w, 1, 1)
    enc2.dispatchThreads_threadsPerThreadgroup_(grid2, tg2)
    enc2.endEncoding()

    cmd.commit()
    cmd.waitUntilCompleted()

    out = Buffer(id=out_id, device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=out_mtl), meta={"n": n, "omega": omega, "shift": shift})
    return {"outputs": {out_id: out}}

_TWIDDLES_CACHE = {}

def _get_stockham_twiddles(n: int, omega: int, rt: Any) -> Any:
    key = (n, omega)
    if key in _TWIDDLES_CACHE:
        return _TWIDDLES_CACHE[key]

    import array
    import math
    p = int(FR_MODULUS)
    r = pow(2, 256, p)
    logn = int(math.log2(n))

    flat: list[int] = []
    for s in range(logn):
        half_len = 1 << s
        step = n // (2 * half_len)
        wlen = pow(int(omega) % p, step, p)
        w = 1
        for k in range(half_len):
            wm = (w * r) % p
            flat.append(int(wm & ((1 << 64) - 1)))
            flat.append(int((wm >> 64) & ((1 << 64) - 1)))
            flat.append(int((wm >> 128) & ((1 << 64) - 1)))
            flat.append(int((wm >> 192) & ((1 << 64) - 1)))
            w = (w * wlen) % p
            
    arr = array.array("Q", flat)
    b = arr.tobytes()
    wlen_buf = rt.device.newBufferWithBytes_length_options_(b, len(b), 0)
    if wlen_buf is None:
        raise RuntimeError("failed to allocate wlen buffer")
    _TWIDDLES_CACHE[key] = wlen_buf
    return wlen_buf

def _roots_evals_from_coeffs_v2(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import array
    import math
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    n = int(inp.data.n) if isinstance(inp.data, MetalBuffer) else int(ctx["attrs"].get("n", 0))
    omega = int(ctx["attrs"]["omega"])
    c = ctx["context"]
    rt = c.metal

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    logn = int(math.log2(n))

    p = int(FR_MODULUS)
    r = pow(2, 256, p)

    wlen_buf = _get_stockham_twiddles(n, omega, rt)

    # Allocate ping-pong buffers
    buf_a = _alloc_metal(ctx, rt, n * 32)
    buf_b = _alloc_metal(ctx, rt, n * 32)

    nb = struct.pack("I", int(n))
    in_size_b = struct.pack("I", int(inp.data.n))

    cmd = rt.queue.commandBuffer()

    # Step 0: Copy & Pad input to buf_a
    # We use poly_scale_shift with shift=1
    shift_bytes = array.array("Q", [int(r & ((1 << 64) - 1)), int((r >> 64) & ((1 << 64) - 1)), int((r >> 128) & ((1 << 64) - 1)), int((r >> 192) & ((1 << 64) - 1))]).tobytes()
    enc0 = cmd.computeCommandEncoder()
    enc0.setComputePipelineState_(rt.pso_poly_scale_shift_fr_mont)
    enc0.setBuffer_offset_atIndex_(inp.data.mtl_buffer, 0, 0)
    enc0.setBuffer_offset_atIndex_(buf_a, 0, 1)
    enc0.setBytes_length_atIndex_(shift_bytes, len(shift_bytes), 2)
    enc0.setBytes_length_atIndex_(nb, len(nb), 3)
    enc0.setBytes_length_atIndex_(in_size_b, len(in_size_b), 4)
    w0 = int(rt.pso_poly_scale_shift_fr_mont.threadExecutionWidth())
    if w0 <= 0: w0 = 64
    tg0 = Metal.MTLSizeMake(w0, 1, 1)
    grid0 = Metal.MTLSizeMake((n + w0 - 1) // w0 * w0, 1, 1)
    enc0.dispatchThreads_threadsPerThreadgroup_(grid0, tg0)
    enc0.endEncoding()

    # Step 1: Stockham Butterfly Operations
    curr_in = buf_a
    curr_out = buf_b
    
    for s in range(logn):
        enc1 = cmd.computeCommandEncoder()
        enc1.setComputePipelineState_(rt.pso_ntt_stockham)
        enc1.setBuffer_offset_atIndex_(curr_in, 0, 0)
        enc1.setBuffer_offset_atIndex_(curr_out, 0, 1)
        enc1.setBuffer_offset_atIndex_(wlen_buf, 0, 2)
        enc1.setBytes_length_atIndex_(nb, len(nb), 3)
        sb = struct.pack("I", int(s))
        enc1.setBytes_length_atIndex_(sb, len(sb), 4)
        
        num_threads = n // 2
        w1 = int(rt.pso_ntt_stockham.threadExecutionWidth())
        if w1 <= 0: w1 = 64
        tg1 = Metal.MTLSizeMake(w1, 1, 1)
        grid1 = Metal.MTLSizeMake((num_threads + w1 - 1) // w1 * w1, 1, 1)
        enc1.dispatchThreads_threadsPerThreadgroup_(grid1, tg1)
        enc1.endEncoding()
        
        # swap buffers
        curr_in, curr_out = curr_out, curr_in

    cmd.commit()
    cmd.waitUntilCompleted()

    # After logn iterations, the result is in curr_in (because we swapped at the end)
    # The other buffer can be discarded. We just wrap curr_in in MetalBuffer.
    out = Buffer(id=out_id, device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=curr_in), meta={"n": n, "omega": omega})
    return {"outputs": {out_id: out}}

def _roots_evals_from_coeffs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import array
    import math
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    n = int(inp.data.n) if isinstance(inp.data, MetalBuffer) else int(ctx["attrs"].get("n", 0))
    omega = int(ctx["attrs"]["omega"])
    if inp.dtype != DType.FR:
        raise ValueError("metal roots_evals_from_coeffs supports FR only")
    if not isinstance(inp.data, MetalBuffer):
        raise ValueError("metal roots_evals_from_coeffs expects MetalBuffer")
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal roots_evals_from_coeffs requires MetalContext with metal runtime")
    
    ntt_mode = getattr(c.config, "metal_ntt_mode", "v1") if c.config else "v1"
    if ntt_mode == "v2":
        return _roots_evals_from_coeffs_v2(ctx)
        
    rt = c.metal

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    logn = int(math.log2(n))
    if (1 << logn) != n:
        raise ValueError("n must be a power of two")

    p = int(FR_MODULUS)
    r = pow(2, 256, p)

    flat: list[int] = []
    length = 2
    # 循环计算每一层的旋转因子
    for _ in range(logn):
        wlen = pow(int(omega) % p, n // length, p)
        wm = (wlen * r) % p
        flat.append(int(wm & ((1 << 64) - 1)))
        flat.append(int((wm >> 64) & ((1 << 64) - 1)))
        flat.append(int((wm >> 128) & ((1 << 64) - 1)))
        flat.append(int((wm >> 192) & ((1 << 64) - 1)))
        length <<= 1
    arr = array.array("Q", flat)
    b = arr.tobytes()
    wlen_buf = rt.device.newBufferWithBytes_length_options_(b, len(b), 0) # 将cpu上计算好的旋转因子拷贝到GPU buffer
    if wlen_buf is None:
        raise RuntimeError("failed to allocate wlen buffer")

    out_mtl = _alloc_metal(ctx, rt, n * 32)
    if out_mtl is None:
        raise RuntimeError("failed to allocate output buffer")

    nb = struct.pack("I", int(n))
    in_size_b = struct.pack("I", int(inp.data.n))

    cmd = rt.queue.commandBuffer()
    
    # 1. Bit-reversal Permutation
    enc1 = cmd.computeCommandEncoder()
    enc1.setComputePipelineState_(rt.pso_ntt_bit_reverse)
    enc1.setBuffer_offset_atIndex_(inp.data.mtl_buffer, 0, 0)
    enc1.setBuffer_offset_atIndex_(out_mtl, 0, 1)
    enc1.setBytes_length_atIndex_(nb, len(nb), 2)
    enc1.setBytes_length_atIndex_(in_size_b, len(in_size_b), 3)
    
    w1 = int(rt.pso_ntt_bit_reverse.threadExecutionWidth())
    if w1 <= 0: w1 = 64
    tg1 = Metal.MTLSizeMake(w1, 1, 1)
    grid1 = Metal.MTLSizeMake((n + w1 - 1) // w1 * w1, 1, 1)
    enc1.dispatchThreads_threadsPerThreadgroup_(grid1, tg1)
    enc1.endEncoding()

    # 2. Butterfly operations layer by layer
    for s in range(logn):
        enc2 = cmd.computeCommandEncoder()
        enc2.setComputePipelineState_(rt.pso_ntt_butterfly)
        enc2.setBuffer_offset_atIndex_(out_mtl, 0, 0)
        enc2.setBuffer_offset_atIndex_(wlen_buf, 0, 1)
        enc2.setBytes_length_atIndex_(nb, len(nb), 2)
        sb = struct.pack("I", int(s))
        enc2.setBytes_length_atIndex_(sb, len(sb), 3)
        
        # 每个 butterfly 操作处理两个元素，所以线程数是 n/2
        num_threads = n // 2
        w2 = int(rt.pso_ntt_butterfly.threadExecutionWidth())
        if w2 <= 0: w2 = 64
        tg2 = Metal.MTLSizeMake(w2, 1, 1)
        grid2 = Metal.MTLSizeMake((num_threads + w2 - 1) // w2 * w2, 1, 1)
        enc2.dispatchThreads_threadsPerThreadgroup_(grid2, tg2)
        enc2.endEncoding()

    cmd.commit()
    cmd.waitUntilCompleted()

    out = Buffer(id=out_id, device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=out_mtl), meta={"n": n, "omega": omega})
    return {"outputs": {out_id: out}}


def _roots_coeffs_from_evals_v2(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import array
    import math
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    n = int(inp.data.n) if isinstance(inp.data, MetalBuffer) else int(ctx["attrs"].get("n", 0))
    omega = int(ctx["attrs"]["omega"])
    c = ctx["context"]
    rt = c.metal

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    logn = int(math.log2(n))

    p = int(FR_MODULUS)
    r = pow(2, 256, p)
    inv_omega = pow(int(omega) % p, -1, p)
    inv_n = pow(int(n), -1, p)
    inv_n_m = (inv_n * r) % p

    wlen_buf = _get_stockham_twiddles(n, inv_omega, rt)

    buf_a = _alloc_metal(ctx, rt, n * 32)
    buf_b = _alloc_metal(ctx, rt, n * 32)

    nb = struct.pack("I", int(n))
    in_size_b = struct.pack("I", int(inp.data.n))
    invn = array.array(
        "Q",
        [
            int(inv_n_m & ((1 << 64) - 1)),
            int((inv_n_m >> 64) & ((1 << 64) - 1)),
            int((inv_n_m >> 128) & ((1 << 64) - 1)),
            int((inv_n_m >> 192) & ((1 << 64) - 1)),
        ],
    ).tobytes()

    cmd = rt.queue.commandBuffer()

    # Step 0: Copy & Pad
    shift_bytes = array.array("Q", [int(r & ((1 << 64) - 1)), int((r >> 64) & ((1 << 64) - 1)), int((r >> 128) & ((1 << 64) - 1)), int((r >> 192) & ((1 << 64) - 1))]).tobytes()
    enc0 = cmd.computeCommandEncoder()
    enc0.setComputePipelineState_(rt.pso_poly_scale_shift_fr_mont)
    enc0.setBuffer_offset_atIndex_(inp.data.mtl_buffer, 0, 0)
    enc0.setBuffer_offset_atIndex_(buf_a, 0, 1)
    enc0.setBytes_length_atIndex_(shift_bytes, len(shift_bytes), 2)
    enc0.setBytes_length_atIndex_(nb, len(nb), 3)
    enc0.setBytes_length_atIndex_(in_size_b, len(in_size_b), 4)
    w0 = int(rt.pso_poly_scale_shift_fr_mont.threadExecutionWidth())
    if w0 <= 0: w0 = 64
    tg0 = Metal.MTLSizeMake(w0, 1, 1)
    grid0 = Metal.MTLSizeMake((n + w0 - 1) // w0 * w0, 1, 1)
    enc0.dispatchThreads_threadsPerThreadgroup_(grid0, tg0)
    enc0.endEncoding()

    # Step 1: Stockham Butterfly Operations
    curr_in = buf_a
    curr_out = buf_b
    
    for s in range(logn):
        enc1 = cmd.computeCommandEncoder()
        enc1.setComputePipelineState_(rt.pso_ntt_stockham)
        enc1.setBuffer_offset_atIndex_(curr_in, 0, 0)
        enc1.setBuffer_offset_atIndex_(curr_out, 0, 1)
        enc1.setBuffer_offset_atIndex_(wlen_buf, 0, 2)
        enc1.setBytes_length_atIndex_(nb, len(nb), 3)
        sb = struct.pack("I", int(s))
        enc1.setBytes_length_atIndex_(sb, len(sb), 4)
        
        num_threads = n // 2
        w1 = int(rt.pso_ntt_stockham.threadExecutionWidth())
        if w1 <= 0: w1 = 64
        tg1 = Metal.MTLSizeMake(w1, 1, 1)
        grid1 = Metal.MTLSizeMake((num_threads + w1 - 1) // w1 * w1, 1, 1)
        enc1.dispatchThreads_threadsPerThreadgroup_(grid1, tg1)
        enc1.endEncoding()
        
        curr_in, curr_out = curr_out, curr_in

    # Step 2: Multiply by inv_n
    enc3 = cmd.computeCommandEncoder()
    enc3.setComputePipelineState_(rt.pso_intt_mul_inv_n)
    enc3.setBuffer_offset_atIndex_(curr_in, 0, 0)
    enc3.setBytes_length_atIndex_(invn, len(invn), 1)
    enc3.setBytes_length_atIndex_(nb, len(nb), 2)
    
    w3 = int(rt.pso_intt_mul_inv_n.threadExecutionWidth())
    if w3 <= 0: w3 = 64
    tg3 = Metal.MTLSizeMake(w3, 1, 1)
    grid3 = Metal.MTLSizeMake((n + w3 - 1) // w3 * w3, 1, 1)
    enc3.dispatchThreads_threadsPerThreadgroup_(grid3, tg3)
    enc3.endEncoding()

    cmd.commit()
    cmd.waitUntilCompleted()

    out = Buffer(id=out_id, device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=curr_in), meta={"n": n, "omega": omega})
    return {"outputs": {out_id: out}}

def _roots_coeffs_from_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import array
    import math
    import Metal  # type: ignore
    import struct

    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    n = int(inp.data.n) if isinstance(inp.data, MetalBuffer) else int(ctx["attrs"].get("n", 0))
    omega = int(ctx["attrs"]["omega"])
    if inp.dtype != DType.FR:
        raise ValueError("metal roots_coeffs_from_evals supports FR only")
    if not isinstance(inp.data, MetalBuffer):
        raise ValueError("metal roots_coeffs_from_evals expects MetalBuffer")
    c = ctx.get("context")
    if c is None or getattr(c, "metal", None) is None:
        raise RuntimeError("metal roots_coeffs_from_evals requires MetalContext with metal runtime")
    
    ntt_mode = getattr(c.config, "metal_ntt_mode", "v1") if c.config else "v1"
    if ntt_mode == "v2":
        return _roots_coeffs_from_evals_v2(ctx)

    rt = c.metal

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two")
    logn = int(math.log2(n))
    if (1 << logn) != n:
        raise ValueError("n must be a power of two")

    p = int(FR_MODULUS)
    r = pow(2, 256, p)
    inv_omega = pow(int(omega) % p, -1, p)
    inv_n = pow(int(n), -1, p)
    inv_n_m = (inv_n * r) % p

    flat: list[int] = []
    length = 2
    for _ in range(logn):
        wlen = pow(int(inv_omega) % p, n // length, p)
        wm = (wlen * r) % p
        flat.append(int(wm & ((1 << 64) - 1)))
        flat.append(int((wm >> 64) & ((1 << 64) - 1)))
        flat.append(int((wm >> 128) & ((1 << 64) - 1)))
        flat.append(int((wm >> 192) & ((1 << 64) - 1)))
        length <<= 1
    arr = array.array("Q", flat)
    b = arr.tobytes()
    wlen_buf = rt.device.newBufferWithBytes_length_options_(b, len(b), 0)
    if wlen_buf is None:
        raise RuntimeError("failed to allocate wlen buffer")

    out_mtl = _alloc_metal(ctx, rt, n * 32)
    if out_mtl is None:
        raise RuntimeError("failed to allocate output buffer")

    nb = struct.pack("I", int(n))
    lb = struct.pack("I", int(logn))
    in_size_b = struct.pack("I", int(inp.data.n))
    invn = array.array(
        "Q",
        [
            int(inv_n_m & ((1 << 64) - 1)),
            int((inv_n_m >> 64) & ((1 << 64) - 1)),
            int((inv_n_m >> 128) & ((1 << 64) - 1)),
            int((inv_n_m >> 192) & ((1 << 64) - 1)),
        ],
    ).tobytes()

    cmd = rt.queue.commandBuffer()
    
    # 1. Bit-reversal Permutation
    enc1 = cmd.computeCommandEncoder()
    enc1.setComputePipelineState_(rt.pso_ntt_bit_reverse)
    enc1.setBuffer_offset_atIndex_(inp.data.mtl_buffer, 0, 0)
    enc1.setBuffer_offset_atIndex_(out_mtl, 0, 1)
    enc1.setBytes_length_atIndex_(nb, len(nb), 2)
    enc1.setBytes_length_atIndex_(in_size_b, len(in_size_b), 3)
    
    w1 = int(rt.pso_ntt_bit_reverse.threadExecutionWidth())
    if w1 <= 0: w1 = 64
    tg1 = Metal.MTLSizeMake(w1, 1, 1)
    grid1 = Metal.MTLSizeMake((n + w1 - 1) // w1 * w1, 1, 1)
    enc1.dispatchThreads_threadsPerThreadgroup_(grid1, tg1)
    enc1.endEncoding()

    # 2. Butterfly operations layer by layer
    for s in range(logn):
        enc2 = cmd.computeCommandEncoder()
        enc2.setComputePipelineState_(rt.pso_ntt_butterfly)
        enc2.setBuffer_offset_atIndex_(out_mtl, 0, 0)
        enc2.setBuffer_offset_atIndex_(wlen_buf, 0, 1)
        enc2.setBytes_length_atIndex_(nb, len(nb), 2)
        sb = struct.pack("I", int(s))
        enc2.setBytes_length_atIndex_(sb, len(sb), 3)
        
        num_threads = n // 2
        w2 = int(rt.pso_ntt_butterfly.threadExecutionWidth())
        if w2 <= 0: w2 = 64
        tg2 = Metal.MTLSizeMake(w2, 1, 1)
        grid2 = Metal.MTLSizeMake((num_threads + w2 - 1) // w2 * w2, 1, 1)
        enc2.dispatchThreads_threadsPerThreadgroup_(grid2, tg2)
        enc2.endEncoding()

    # 3. Multiply by inv_n
    enc3 = cmd.computeCommandEncoder()
    enc3.setComputePipelineState_(rt.pso_intt_mul_inv_n)
    enc3.setBuffer_offset_atIndex_(out_mtl, 0, 0)
    enc3.setBytes_length_atIndex_(invn, len(invn), 1)
    enc3.setBytes_length_atIndex_(nb, len(nb), 2)
    
    w3 = int(rt.pso_intt_mul_inv_n.threadExecutionWidth())
    if w3 <= 0: w3 = 64
    tg3 = Metal.MTLSizeMake(w3, 1, 1)
    grid3 = Metal.MTLSizeMake((n + w3 - 1) // w3 * w3, 1, 1)
    enc3.dispatchThreads_threadsPerThreadgroup_(grid3, tg3)
    enc3.endEncoding()

    cmd.commit()
    cmd.waitUntilCompleted()

    out = Buffer(id=out_id, device=Device.METAL, dtype=DType.FR, data=MetalBuffer(dtype="fr_mont_u64x4", n=n, mtl_buffer=out_mtl), meta={"n": n, "omega": omega})
    return {"outputs": {out_id: out}}
