import math
import struct
import array
from typing import Any, List, Sequence, Tuple
from pyZKP.common.crypto.ecc.bn254 import G1, G1_ZERO, g1_add, g1_mul
from pyZKP.common.crypto.field.fr import FR_MODULUS

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

