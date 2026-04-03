import math
import struct
import array
from typing import Any, List, Sequence, Tuple
from crypto.ecc.bn254 import G1, G1_ZERO, g1_add, g1_mul
from crypto.field.fr import FR_MODULUS

def _fq_to_mont(val: int) -> int:
    P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
    R = pow(2, 256, P)
    return (val * R) % P

def _pack_g1_point(p: G1) -> bytes:
    # p is (x, y, z) in affine or projective. py_ecc uses (x, y, z).
    # We normalize to affine (Z=1) so that it trivially matches Jacobian coordinates.
    from py_ecc.optimized_bn128 import normalize
    P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
    if p[2] == 0:
        # 无穷远点
        return array.array("Q", [0]*12).tobytes()
    
    p_norm = normalize(p)
    
    # 转为 Montgomery
    x_m = _fq_to_mont(int(p_norm[0]) % P)
    y_m = _fq_to_mont(int(p_norm[1]) % P)
    z_m = _fq_to_mont(1)

    flat = [
        int(x_m & ((1<<64)-1)), int((x_m>>64) & ((1<<64)-1)), int((x_m>>128) & ((1<<64)-1)), int((x_m>>192) & ((1<<64)-1)),
        int(y_m & ((1<<64)-1)), int((y_m>>64) & ((1<<64)-1)), int((y_m>>128) & ((1<<64)-1)), int((y_m>>192) & ((1<<64)-1)),
        int(z_m & ((1<<64)-1)), int((z_m>>64) & ((1<<64)-1)), int((z_m>>128) & ((1<<64)-1)), int((z_m>>192) & ((1<<64)-1)),
    ]
    return array.array("Q", flat).tobytes()

def _unpack_g1_point(b: bytes, offset: int) -> G1:
    from py_ecc.optimized_bn128 import FQ
    arr = array.array("Q")
    arr.frombytes(b[offset:offset+96])
    
    x_m = arr[0] | (arr[1]<<64) | (arr[2]<<128) | (arr[3]<<192)
    y_m = arr[4] | (arr[5]<<64) | (arr[6]<<128) | (arr[7]<<192)
    z_m = arr[8] | (arr[9]<<64) | (arr[10]<<128) | (arr[11]<<192)
    
    if z_m == 0:
        return G1_ZERO
        
    P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
    inv_R = pow(pow(2, 256, P), -1, P)
    
    X = (x_m * inv_R) % P
    Y = (y_m * inv_R) % P
    Z = (z_m * inv_R) % P
    
    # Convert from Jacobian to Affine
    invZ = pow(Z, -1, P)
    invZ2 = (invZ * invZ) % P
    invZ3 = (invZ2 * invZ) % P
    
    x = (X * invZ2) % P
    y = (Y * invZ3) % P
    
    return (FQ(x), FQ(y), FQ(1))

def metal_msm_g1(rt: Any, points: Sequence[G1], scalars_mtl_buffer: Any, num_points: int) -> G1:
    import Metal
    
    if num_points == 0:
        return G1_ZERO
        
    # 动态计算 window_bits
    window_bits = max(4, min(16, int(num_points).bit_length() - 2))
    
    max_bits = FR_MODULUS.bit_length() # 254
    window_count = (max_bits + window_bits - 1) // window_bits
    buckets_per_window = 1 << window_bits
    total_buckets = window_count * buckets_per_window
    
    # 1. 准备 points buffer
    # 在实际应用中，points (如 SRS) 是固定的，应该预先缓存在显存中。
    # 这里为了简便先做 CPU -> GPU 的拷贝，未来可以进一步优化。
    pts_bytes = bytearray()
    for p in points:
        pts_bytes.extend(_pack_g1_point(p))
    pts_buf = rt.device.newBufferWithBytes_length_options_(pts_bytes, len(pts_bytes), 0)
    
    # 2. 分配 buckets buffer 和 window_sums buffer
    # 注意：MetalBuffer 默认不一定是全 0，我们需要用 0 初始化，因为 0 代表无穷远点 (Z=0)
    zeros_buckets = bytes(total_buckets * 96)
    buckets_buf = rt.device.newBufferWithBytes_length_options_(zeros_buckets, len(zeros_buckets), 0)
    
    zeros_sums = bytes(window_count * 96)
    window_sums_buf = rt.device.newBufferWithBytes_length_options_(zeros_sums, len(zeros_sums), 0)
    
    cmd = rt.queue.commandBuffer()
    
    # Phase 1: Bucket Accumulate
    enc1 = cmd.computeCommandEncoder()
    enc1.setComputePipelineState_(rt.pso_msm_bucket_accumulate)
    enc1.setBuffer_offset_atIndex_(pts_buf, 0, 0)
    enc1.setBuffer_offset_atIndex_(scalars_mtl_buffer, 0, 1)
    enc1.setBuffer_offset_atIndex_(buckets_buf, 0, 2)
    enc1.setBytes_length_atIndex_(struct.pack("I", num_points), 4, 3)
    enc1.setBytes_length_atIndex_(struct.pack("I", window_bits), 4, 4)
    
    # 线程网格：总 bucket 数量
    w1 = int(rt.pso_msm_bucket_accumulate.threadExecutionWidth())
    if w1 <= 0: w1 = 64
    tg1 = Metal.MTLSizeMake(w1, 1, 1)
    grid1 = Metal.MTLSizeMake((total_buckets + w1 - 1) // w1 * w1, 1, 1)
    enc1.dispatchThreads_threadsPerThreadgroup_(grid1, tg1)
    enc1.endEncoding()
    
    # Phase 2: Bucket Reduce (每个 window 一个线程)
    enc2 = cmd.computeCommandEncoder()
    enc2.setComputePipelineState_(rt.pso_msm_bucket_reduce)
    enc2.setBuffer_offset_atIndex_(buckets_buf, 0, 0)
    enc2.setBuffer_offset_atIndex_(window_sums_buf, 0, 1)
    enc2.setBytes_length_atIndex_(struct.pack("I", window_bits), 4, 2)
    
    w2 = int(rt.pso_msm_bucket_reduce.threadExecutionWidth())
    if w2 <= 0: w2 = 64
    tg2 = Metal.MTLSizeMake(w2, 1, 1)
    grid2 = Metal.MTLSizeMake((window_count + w2 - 1) // w2 * w2, 1, 1)
    enc2.dispatchThreads_threadsPerThreadgroup_(grid2, tg2)
    enc2.endEncoding()
    
    cmd.commit()
    cmd.waitUntilCompleted()
    
    # Phase 3: Final Accumulation on CPU
    # 这一步计算量极小 (几十次点加)，直接在 CPU 上完成。
    window_sums_bytes = window_sums_buf.contents().as_buffer(window_count * 96)
    
    acc = G1_ZERO
    for i in reversed(range(window_count)):
        w_sum = _unpack_g1_point(window_sums_bytes, i * 96)
        # acc = acc * (2^window_bits)
        if i != window_count - 1:
            acc = g1_mul(acc, 1 << window_bits)
        acc = g1_add(acc, w_sum)
        
    return acc

def _encode_signed_digits(scalars: Sequence[int], window_bits: int, window_count: int) -> bytearray:
    """
    将标量数组转换为 Signed-Digit (w-NAF 变体) 格式的字节流。
    每个窗口的值被编码为一个 32 位有符号整数 (int32)。
    返回的 bytearray 结构为 [scalar0_w0, scalar0_w1... scalar1_w0, scalar1_w1...]
    """
    import struct
    res = bytearray()
    half = 1 << (window_bits - 1)
    mask = (1 << window_bits) - 1
    
    for s in scalars:
        # 复制一份用于移位
        temp_s = s
        carry = 0
        for w in range(window_count):
            # 提取当前窗口的值并加上之前的进位
            val = (temp_s & mask) + carry
            temp_s >>= window_bits
            
            # 如果值 >= 2^(w-1)，则变为负数，并产生向高位的进位
            if val >= half:
                val_signed = val - (1 << window_bits)
                carry = 1
            else:
                val_signed = val
                carry = 0
                
            res.extend(struct.pack("i", val_signed))
            
        # 注意：如果最高位还有 carry=1，说明标量超出了 window_count 的表示范围。
        # 对于 BN254，标量最大 254 bits。只要 window_count * window_bits >= 255，就不会溢出。
    return res

def metal_msm_g1_v2(rt: Any, points: Sequence[G1], scalars_mtl_buffer: Any, num_points: int) -> G1:
    """
    V2 优化版本：使用 Signed-Digit (w-NAF) 编码，将 Bucket 数量减半。
    注意：这里的 scalars_mtl_buffer 预期是已经经过 _encode_signed_digits 转换的 int32 数组。
    如果传入的是原始的 Montgomery 形式，需要在此处拦截并转换。
    为了兼容之前的调用签名，我们在这里做个 Hack：如果传入的是原始 Buffer，我们在 Python 端转换。
    """
    import Metal
    
    if num_points == 0:
        return G1_ZERO
        
    window_bits = max(4, min(16, int(num_points).bit_length() - 2))
    max_bits = FR_MODULUS.bit_length() # 254
    # 确保覆盖 255 bits 以容纳最高位可能的进位
    window_count = (max_bits + 1 + window_bits - 1) // window_bits
    
    # V2 优化：bucket 数量减半。注意绝对值最大为 2^(w-1)，所以需要 +1 个桶（0 不用，1~half）
    buckets_per_window = (1 << (window_bits - 1)) + 1
    total_buckets = window_count * buckets_per_window
    
    # 1. 准备 points
    pts_bytes = bytearray()
    for p in points:
        pts_bytes.extend(_pack_g1_point(p))
    pts_buf = rt.device.newBufferWithBytes_length_options_(pts_bytes, len(pts_bytes), 0)
    
    # 2. 准备 scalars (检查是否需要转换)
    # 在 compare_msm_v1_v2.py 中，传入的是原始 Montgomery 形式的 raw bytes。
    # 为了纯粹测试 GPU 性能，我们在实际工程中会在外层预先转换好。
    # 这里我们直接把 scalars_mtl_buffer 当作 raw_bytes 读出来，在 CPU 转换后再传给 GPU。
    # （这在 A/B 测试中会包含 CPU 转换时间，但这是公平的，因为这是预处理代价）。
    raw_scalars_bytes = scalars_mtl_buffer.contents().as_buffer(num_points * 32)
    import struct
    _FR_P = FR_MODULUS
    _FR_R_INV = pow(pow(2, 256, _FR_P), -1, _FR_P)
    
    scalars_int = []
    for i in range(num_points):
        # 读取 4 个 uint64
        v0, v1, v2, v3 = struct.unpack_from("QQQQ", raw_scalars_bytes, i * 32)
        val_mont = v0 | (v1 << 64) | (v2 << 128) | (v3 << 192)
        # 转回普通形式
        val_std = (val_mont * _FR_R_INV) % _FR_P
        scalars_int.append(val_std)
        
    signed_bytes = _encode_signed_digits(scalars_int, window_bits, window_count)
    signed_scalars_buf = rt.device.newBufferWithBytes_length_options_(signed_bytes, len(signed_bytes), 0)
    
    # 3. CSR Buffers
    zeros_offsets = bytes(total_buckets * 4)
    bucket_offsets_buf = rt.device.newBufferWithBytes_length_options_(zeros_offsets, len(zeros_offsets), 0)
    
    row_pointers_buf = rt.device.newBufferWithLength_options_((total_buckets + 1) * 4, 0)
    sorted_indices_buf = rt.device.newBufferWithLength_options_(num_points * window_count * 4, 0)
    
    zeros_buckets = bytes(total_buckets * 96)
    buckets_buf = rt.device.newBufferWithBytes_length_options_(zeros_buckets, len(zeros_buckets), 0)
    
    zeros_sums = bytes(window_count * 96)
    window_sums_buf = rt.device.newBufferWithBytes_length_options_(zeros_sums, len(zeros_sums), 0)
    
    cmd = rt.queue.commandBuffer()
    
    # --- Step 1: Histogram ---
    enc_hist = cmd.computeCommandEncoder()
    enc_hist.setComputePipelineState_(rt.pso_msm_csr_histogram)
    enc_hist.setBuffer_offset_atIndex_(signed_scalars_buf, 0, 0)
    enc_hist.setBuffer_offset_atIndex_(bucket_offsets_buf, 0, 1)
    enc_hist.setBytes_length_atIndex_(struct.pack("I", num_points), 4, 2)
    enc_hist.setBytes_length_atIndex_(struct.pack("I", window_count), 4, 3)
    enc_hist.setBytes_length_atIndex_(struct.pack("I", buckets_per_window), 4, 4)
    w_hist = int(rt.pso_msm_csr_histogram.threadExecutionWidth())
    if w_hist <= 0: w_hist = 64
    enc_hist.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake((num_points + w_hist - 1) // w_hist * w_hist, 1, 1),
        Metal.MTLSizeMake(w_hist, 1, 1)
    )
    enc_hist.endEncoding()
    
    # --- Step 2: Prefix Sum ---
    enc_pre = cmd.computeCommandEncoder()
    enc_pre.setComputePipelineState_(rt.pso_msm_csr_prefix_sum)
    enc_pre.setBuffer_offset_atIndex_(bucket_offsets_buf, 0, 0)
    enc_pre.setBuffer_offset_atIndex_(row_pointers_buf, 0, 1)
    enc_pre.setBytes_length_atIndex_(struct.pack("I", total_buckets), 4, 2)
    enc_pre.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(1, 1, 1)
    )
    enc_pre.endEncoding()
    
    # --- Step 3: Scatter ---
    enc_scat = cmd.computeCommandEncoder()
    enc_scat.setComputePipelineState_(rt.pso_msm_csr_scatter)
    enc_scat.setBuffer_offset_atIndex_(signed_scalars_buf, 0, 0)
    enc_scat.setBuffer_offset_atIndex_(bucket_offsets_buf, 0, 1)
    enc_scat.setBuffer_offset_atIndex_(sorted_indices_buf, 0, 2)
    enc_scat.setBytes_length_atIndex_(struct.pack("I", num_points), 4, 3)
    enc_scat.setBytes_length_atIndex_(struct.pack("I", window_count), 4, 4)
    enc_scat.setBytes_length_atIndex_(struct.pack("I", buckets_per_window), 4, 5)
    w_scat = int(rt.pso_msm_csr_scatter.threadExecutionWidth())
    if w_scat <= 0: w_scat = 64
    enc_scat.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake((num_points + w_scat - 1) // w_scat * w_scat, 1, 1),
        Metal.MTLSizeMake(w_scat, 1, 1)
    )
    enc_scat.endEncoding()
    
    # --- Step 4: Accumulate ---
    enc1 = cmd.computeCommandEncoder()
    enc1.setComputePipelineState_(rt.pso_msm_bucket_accumulate_v2)
    enc1.setBuffer_offset_atIndex_(pts_buf, 0, 0)
    enc1.setBuffer_offset_atIndex_(sorted_indices_buf, 0, 1)
    enc1.setBuffer_offset_atIndex_(row_pointers_buf, 0, 2)
    enc1.setBuffer_offset_atIndex_(buckets_buf, 0, 3)
    enc1.setBytes_length_atIndex_(struct.pack("I", window_count), 4, 4)
    enc1.setBytes_length_atIndex_(struct.pack("I", buckets_per_window), 4, 5)
    w1 = int(rt.pso_msm_bucket_accumulate_v2.threadExecutionWidth())
    if w1 <= 0: w1 = 64
    enc1.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake((total_buckets + w1 - 1) // w1 * w1, 1, 1),
        Metal.MTLSizeMake(w1, 1, 1)
    )
    enc1.endEncoding()
    
    # --- Step 5: Reduce ---
    enc2 = cmd.computeCommandEncoder()
    enc2.setComputePipelineState_(rt.pso_msm_bucket_reduce_v2)
    enc2.setBuffer_offset_atIndex_(buckets_buf, 0, 0)
    enc2.setBuffer_offset_atIndex_(window_sums_buf, 0, 1)
    enc2.setBytes_length_atIndex_(struct.pack("I", window_bits), 4, 2)
    
    w2 = int(rt.pso_msm_bucket_reduce_v2.threadExecutionWidth())
    if w2 <= 0: w2 = 64
    tg2 = Metal.MTLSizeMake(w2, 1, 1)
    grid2 = Metal.MTLSizeMake((window_count + w2 - 1) // w2 * w2, 1, 1)
    enc2.dispatchThreads_threadsPerThreadgroup_(grid2, tg2)
    enc2.endEncoding()
    
    cmd.commit()
    cmd.waitUntilCompleted()
    
    # Phase 3: Final Accumulation on CPU
    window_sums_bytes = window_sums_buf.contents().as_buffer(window_count * 96)
    
    acc = G1_ZERO
    for i in reversed(range(window_count)):
        w_sum = _unpack_g1_point(window_sums_bytes, i * 96)
        if i != window_count - 1:
            acc = g1_mul(acc, 1 << window_bits)
        acc = g1_add(acc, w_sum)
        
    return acc

