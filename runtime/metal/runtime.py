from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from runtime.metal.ecc_source import _ECC_KERNEL_SOURCE

# 检查 Metal 是否可用

# 导入 Metal 和 Foundation 库
def metal_available() -> bool:
    try:
        import Metal  # type: ignore
        import Foundation  # type: ignore
        return True
    except Exception:
        return False

_KERNEL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// v0: 最低 64 位，v1: 中间 64 位，v2: 高 64 位，v3: 最高 64 位
struct alignas(32) Fr {
    ulong v0;
    ulong v1;
    ulong v2;
    ulong v3;
};

constant ulong FR_MOD0 = 4891460686036598785ul;
constant ulong FR_MOD1 = 2896914383306846353ul;
constant ulong FR_MOD2 = 13281191951274694749ul;
constant ulong FR_MOD3 = 3486998266802970665ul;
constant ulong FR_INV_NEG = 14042775128853446655ul; // BN254 Fr: -P^{-1} mod 2^64

// 蒙哥马利域中的 1 (即 R mod P)
constant ulong FR_R_MOD0 = 12436184717236109307ul;
constant ulong FR_R_MOD1 = 3962172157175319849ul;
constant ulong FR_R_MOD2 = 7381016538464732718ul;
constant ulong FR_R_MOD3 = 1011752739694698287ul;

// 64 位加法，带进位
inline ulong addc64(ulong a, ulong b, thread ulong &carry) {
    ulong s = a + b;
    ulong c1 = (s < a) ? 1ul : 0ul;
    ulong s2 = s + carry;
    ulong c2 = (s2 < s) ? 1ul : 0ul;
    carry = c1 + c2;
    return s2;
}
// 64 位减法，带借位
inline ulong subb64(ulong a, ulong b, thread ulong &borrow) {
    ulong d = a - b;
    ulong b1 = (a < b) ? 1ul : 0ul;
    ulong d2 = d - borrow;
    ulong b2 = (d < borrow) ? 1ul : 0ul;
    borrow = b1 + b2;
    return d2;
}

// 64 位乘加，计算 acc = acc + x * y + carry
// 返回新的 carry
inline void mac64(ulong x, ulong y, thread ulong &acc, thread ulong &carry) {
    ulong lo = x * y;
    ulong hi = mulhi(x, y);
    
    ulong c1 = 0;
    ulong s1 = addc64(acc, lo, c1);
    
    ulong c2 = 0;
    acc = addc64(s1, carry, c2);
    
    carry = hi + c1 + c2;
}

// 判断一个 256 bit 域元素是否大于等于 p
inline bool geq_mod(Fr a) {
    if (a.v3 != FR_MOD3) return a.v3 > FR_MOD3;
    if (a.v2 != FR_MOD2) return a.v2 > FR_MOD2;
    if (a.v1 != FR_MOD1) return a.v1 > FR_MOD1;
    return a.v0 >= FR_MOD0;
}

// 256 bit 域元素减模数
inline Fr sub_mod(Fr a) {
    ulong borrow = 0;
    a.v0 = subb64(a.v0, FR_MOD0, borrow);
    a.v1 = subb64(a.v1, FR_MOD1, borrow);
    a.v2 = subb64(a.v2, FR_MOD2, borrow);
    a.v3 = subb64(a.v3, FR_MOD3, borrow);
    return a;
}

inline Fr add_mod(Fr a, Fr b) {
    ulong carry = 0;
    Fr r;
    r.v0 = addc64(a.v0, b.v0, carry);
    r.v1 = addc64(a.v1, b.v1, carry);
    r.v2 = addc64(a.v2, b.v2, carry);
    r.v3 = addc64(a.v3, b.v3, carry);
    if (carry != 0 || geq_mod(r)) {
        r = sub_mod(r);
    }
    return r;
}

inline Fr sub_mod2(Fr a, Fr b) {
    ulong borrow = 0;
    Fr r;
    r.v0 = subb64(a.v0, b.v0, borrow);
    r.v1 = subb64(a.v1, b.v1, borrow);
    r.v2 = subb64(a.v2, b.v2, borrow);
    r.v3 = subb64(a.v3, b.v3, borrow);
    if (borrow != 0) {
        ulong carry = 0;
        r.v0 = addc64(r.v0, FR_MOD0, carry);
        r.v1 = addc64(r.v1, FR_MOD1, carry);
        r.v2 = addc64(r.v2, FR_MOD2, carry);
        r.v3 = addc64(r.v3, FR_MOD3, carry);
    }
    return r;
}

// 256 bit 域元素乘法，Montgomery 表示法
inline Fr mont_mul(Fr a, Fr b) {
    // 使用数组以确保在内存中的连续分配，避免未定义行为
    ulong t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // 将 256-bit 乘法拆分为 16 个 64-bit 乘加
    {
        ulong carry = 0;
        mac64(a.v0, b.v0, t[0], carry);
        mac64(a.v0, b.v1, t[1], carry);
        mac64(a.v0, b.v2, t[2], carry);
        mac64(a.v0, b.v3, t[3], carry);
        t[4] = carry;
    }
    {
        ulong carry = 0;
        mac64(a.v1, b.v0, t[1], carry);
        mac64(a.v1, b.v1, t[2], carry);
        mac64(a.v1, b.v2, t[3], carry);
        mac64(a.v1, b.v3, t[4], carry);
        t[5] = carry;
    }
    {
        ulong carry = 0;
        mac64(a.v2, b.v0, t[2], carry);
        mac64(a.v2, b.v1, t[3], carry);
        mac64(a.v2, b.v2, t[4], carry);
        mac64(a.v2, b.v3, t[5], carry);
        t[6] = carry;
    }
    {
        ulong carry = 0;
        mac64(a.v3, b.v0, t[3], carry);
        mac64(a.v3, b.v1, t[4], carry);
        mac64(a.v3, b.v2, t[5], carry);
        mac64(a.v3, b.v3, t[6], carry);
        t[7] = carry;
    }

    for (uint i = 0; i < 4; i++) {
        ulong m = t[i] * FR_INV_NEG;

        ulong carry = 0;
        mac64(m, FR_MOD0, t[i+0], carry);
        mac64(m, FR_MOD1, t[i+1], carry);
        mac64(m, FR_MOD2, t[i+2], carry);
        mac64(m, FR_MOD3, t[i+3], carry);

        ulong c = 0;
        t[i+4] = addc64(t[i+4], carry, c);
        uint k = i + 5;
        while (c != 0 && k < 8) {
            t[k] = addc64(t[k], 0ul, c);
            k++;
        }
    }

    Fr r;
    r.v0 = t[4];
    r.v1 = t[5];
    r.v2 = t[6];
    r.v3 = t[7];
    if (geq_mod(r)) {
        r = sub_mod(r);
    }
    return r;
}

inline Fr power(Fr base, uint exp) {
    Fr res;
    res.v0 = FR_R_MOD0; res.v1 = FR_R_MOD1; res.v2 = FR_R_MOD2; res.v3 = FR_R_MOD3;
    Fr cur = base;
    while (exp > 0) {
        if (exp & 1) res = mont_mul(res, cur);
        cur = mont_mul(cur, cur);
        exp >>= 1;
    }
    return res;
}

kernel void pointwise_mul_fr_mont(
    const device Fr* a [[buffer(0)]],
    const device Fr* b [[buffer(1)]],
    device Fr* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = mont_mul(a[gid], b[gid]);
}

kernel void poly_sub_fr_mont(
    const device Fr* a [[buffer(0)]],
    const device Fr* b [[buffer(1)]],
    device Fr* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& n_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    if (gid < n_b) {
        out[gid] = sub_mod2(a[gid], b[gid]);
    } else {
        out[gid] = a[gid];
    }
}

kernel void poly_scale_shift_fr_mont(
    const device Fr* in [[buffer(0)]],
    device Fr* out [[buffer(1)]],
    constant Fr& shift [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& in_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    if (gid < in_size) {
        Fr s = power(shift, gid);
        out[gid] = mont_mul(in[gid], s);
    } else {
        Fr zero;
        zero.v0 = 0; zero.v1 = 0; zero.v2 = 0; zero.v3 = 0;
        out[gid] = zero;
    }
}

// -----------------------------------------------------------------------------
// 并行 NTT/iNTT 实现 (Cooley-Tukey / Gentleman-Sande)
// -----------------------------------------------------------------------------

// 1. Bit-reversal Permutation (每个线程处理一个元素的拷贝和位反转)
kernel void ntt_bit_reverse(
    const device Fr* in [[buffer(0)]],
    device Fr* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& in_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    // Reverse bits of gid
    uint j = 0;
    uint bit = n >> 1;
    uint i = gid;
    while (bit > 0) {
        if (i & 1) j |= bit;
        i >>= 1;
        bit >>= 1;
    }
    
    if (gid < in_size) {
        out[j] = in[gid];
    } else {
        Fr zero;
        zero.v0 = 0; zero.v1 = 0; zero.v2 = 0; zero.v3 = 0;
        out[j] = zero;
    }
}

// 2. NTT Butterfly operations (每个线程处理一对蝶形运算)
kernel void ntt_butterfly(
    device Fr* inout [[buffer(0)]],
    const device Fr* wlen_s [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& s [[buffer(3)]], // 当前层数 (0 to logn - 1)
    uint gid [[thread_position_in_grid]]
) {
    uint length = 2 << s;
    uint half_len = length >> 1;
    
    // 总共有 n / 2 个蝶形操作
    if (gid >= n / 2) return;
    
    // 计算当前线程对应的元素对 (u, v) 的索引
    uint i = (gid / half_len) * length + (gid % half_len);
    uint k = i + half_len;
    
    // 获取当前层的旋转因子
    Fr wlen = wlen_s[s];
    
    // 计算 w = wlen ^ (gid % half_len)
    // 这里为了极致并行，我们可以通过 power 函数或者预先在 CPU/GPU 算好当前层的所有的 w
    // 最简单的方法是预计算：但由于我们这里只传入了 wlen，我们可以简单地做 O(s) 次乘法
    Fr w;
    w.v0 = FR_R_MOD0; w.v1 = FR_R_MOD1; w.v2 = FR_R_MOD2; w.v3 = FR_R_MOD3;
    uint power_idx = gid % half_len;
    
    // 快速幂计算 w = wlen ^ power_idx
    Fr cur = wlen;
    while (power_idx > 0) {
        if (power_idx & 1) w = mont_mul(w, cur);
        cur = mont_mul(cur, cur);
        power_idx >>= 1;
    }
    
    Fr u = inout[i];
    Fr v = mont_mul(inout[k], w);
    
    inout[i] = add_mod(u, v);
    inout[k] = sub_mod2(u, v);
}

// iNTT 乘 inv_n
kernel void intt_mul_inv_n(
    device Fr* inout [[buffer(0)]],
    constant Fr& inv_n [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    inout[gid] = mont_mul(inout[gid], inv_n);
}

// -----------------------------------------------------------------------------
// 并行 NTT/iNTT 实现 v2 (Stockham + 预计算旋转因子)
// -----------------------------------------------------------------------------

kernel void ntt_stockham(
    const device Fr* in_buf [[buffer(0)]],
    device Fr* out_buf [[buffer(1)]],
    const device Fr* twiddles [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& s [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n / 2) return;
    
    uint half_len = 1 << s;
    uint j = gid / half_len;
    uint k = gid % half_len;
    
    uint i0 = j * half_len + k;
    uint i1 = i0 + n / 2;
    
    Fr u = in_buf[i0];
    Fr w = twiddles[(1 << s) - 1 + k];
    Fr v = mont_mul(in_buf[i1], w);
    
    uint out_idx0 = j * (2 * half_len) + k;
    uint out_idx1 = out_idx0 + half_len;
    
    out_buf[out_idx0] = add_mod(u, v);
    out_buf[out_idx1] = sub_mod2(u, v);
}

// 保留原有的单线程版本作备用或兼容（可选），我们将修改 Python 端调用新的并行版本
kernel void roots_evals_from_coeffs_fr_mont(
    const device Fr* in [[buffer(0)]],
    device Fr* out [[buffer(1)]],
    const device Fr* wlen_s [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& logn [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    for (uint i = 0; i < n; i++) {
        out[i] = in[i];
    }
    
    uint j = 0;
    for (uint i = 1; i < n; i++) {
        uint bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            Fr tmp = out[i];
            out[i] = out[j];
            out[j] = tmp;
        }
    }
    
    uint length = 2;
    for (uint s = 0; s < logn; s++) {
        Fr wlen = wlen_s[s];
        uint half_len = length >> 1;
        for (uint i = 0; i < n; i += length) {
            Fr w;
            w.v0 = FR_R_MOD0;
            w.v1 = FR_R_MOD1;
            w.v2 = FR_R_MOD2;
            w.v3 = FR_R_MOD3;
            for (uint k = 0; k < half_len; k++) {
                Fr u = out[i + k];
                Fr v = mont_mul(out[i + k + half_len], w);
                out[i + k] = add_mod(u, v);
                out[i + k + half_len] = sub_mod2(u, v);
                w = mont_mul(w, wlen);
            }
        }
        length <<= 1;
    }
}

kernel void roots_coeffs_from_evals_fr_mont(
    const device Fr* in [[buffer(0)]],
    device Fr* out [[buffer(1)]],
    const device Fr* wlen_s [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& logn [[buffer(4)]],
    constant Fr& inv_n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    for (uint i = 0; i < n; i++) {
        out[i] = in[i];
    }
    uint j = 0;
    for (uint i = 1; i < n; i++) {
        uint bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            Fr tmp = out[i];
            out[i] = out[j];
            out[j] = tmp;
        }
    }
    uint length = 2;
    for (uint s = 0; s < logn; s++) {
        Fr wlen = wlen_s[s];
        uint half_len = length >> 1;
        for (uint i = 0; i < n; i += length) {
            Fr w;
            w.v0 = FR_R_MOD0;
            w.v1 = FR_R_MOD1;
            w.v2 = FR_R_MOD2;
            w.v3 = FR_R_MOD3;
            for (uint k = 0; k < half_len; k++) {
                Fr u = out[i + k];
                Fr v = mont_mul(out[i + k + half_len], w);
                out[i + k] = add_mod(u, v);
                out[i + k + half_len] = sub_mod2(u, v);
                w = mont_mul(w, wlen);
            }
        }
        length <<= 1;
    }
    for (uint i = 0; i < n; i++) {
        out[i] = mont_mul(out[i], inv_n);
    }
}
"""

# Metal 运行时数据类，作为管理 METAL 资源的生命周期容器
@dataclass
class MetalRuntime:
    device: Any # Metal 设备
    queue: Any # Metal 命令队列，用于提交 Metal 命令
    lib: Any # Metal 库，包含 Metal 内核函数的编译结果
    ecc_lib: Any # 包含 ECC 操作的 Metal 库
    pso_pointwise_mul_fr_mont: Any
    pso_poly_sub_fr_mont: Any
    pso_poly_scale_shift_fr_mont: Any
    pso_roots_evals_from_coeffs_fr_mont: Any
    pso_roots_coeffs_from_evals_fr_mont: Any
    pso_msm_bucket_accumulate: Any
    pso_msm_bucket_reduce: Any
    pso_msm_csr_histogram: Any
    pso_msm_csr_prefix_sum: Any
    pso_msm_csr_scatter: Any
    pso_msm_bucket_accumulate_v2: Any
    pso_msm_bucket_reduce_v2: Any
    pso_ntt_bit_reverse: Any
    pso_ntt_butterfly: Any
    pso_intt_mul_inv_n: Any
    pso_ntt_stockham: Any

    # 初始化并且装配所有的 Metal 核心组件
    @staticmethod
    def create_default() -> "MetalRuntime":
        import Metal  # type: ignore
        import Foundation  # type: ignore

        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is None:
            raise RuntimeError("no Metal device")

        err = None
        opts = None
        lib, err = dev.newLibraryWithSource_options_error_(_KERNEL_SOURCE, opts, None)
        if lib is None:
            raise RuntimeError(f"failed to compile Metal library: {err}")
            
        ecc_lib, err = dev.newLibraryWithSource_options_error_(_KERNEL_SOURCE + "\n" + _ECC_KERNEL_SOURCE, opts, None)
        if ecc_lib is None:
            raise RuntimeError(f"failed to compile ECC Metal library: {err}")

        fn0 = lib.newFunctionWithName_("pointwise_mul_fr_mont")
        if fn0 is None:
            raise RuntimeError("failed to find kernel function: pointwise_mul_fr_mont")
        pso0, err = dev.newComputePipelineStateWithFunction_error_(fn0, None)
        if pso0 is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn_sub = lib.newFunctionWithName_("poly_sub_fr_mont")
        if fn_sub is None:
            raise RuntimeError("failed to find kernel function: poly_sub_fr_mont")
        pso_sub, err = dev.newComputePipelineStateWithFunction_error_(fn_sub, None)
        if pso_sub is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn_scale = lib.newFunctionWithName_("poly_scale_shift_fr_mont")
        if fn_scale is None:
            raise RuntimeError("failed to find kernel function: poly_scale_shift_fr_mont")
        pso_scale, err = dev.newComputePipelineStateWithFunction_error_(fn_scale, None)
        if pso_scale is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn1 = lib.newFunctionWithName_("roots_evals_from_coeffs_fr_mont")
        if fn1 is None:
            raise RuntimeError("failed to find kernel function: roots_evals_from_coeffs_fr_mont")
        pso1, err = dev.newComputePipelineStateWithFunction_error_(fn1, None)
        if pso1 is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn2 = lib.newFunctionWithName_("roots_coeffs_from_evals_fr_mont")
        if fn2 is None:
            raise RuntimeError("failed to find kernel function: roots_coeffs_from_evals_fr_mont")
        pso2, err = dev.newComputePipelineStateWithFunction_error_(fn2, None)
        if pso2 is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn_msm1 = ecc_lib.newFunctionWithName_("msm_bucket_accumulate")
        if fn_msm1 is None:
            raise RuntimeError("failed to find kernel function: msm_bucket_accumulate")
        pso_msm1, err = dev.newComputePipelineStateWithFunction_error_(fn_msm1, None)
        if pso_msm1 is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn_msm2 = ecc_lib.newFunctionWithName_("msm_bucket_reduce")
        if fn_msm2 is None:
            raise RuntimeError("failed to find kernel function: msm_bucket_reduce")
        pso_msm2, err = dev.newComputePipelineStateWithFunction_error_(fn_msm2, None)
        if pso_msm2 is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")
            
        fn_csr_hist = ecc_lib.newFunctionWithName_("msm_csr_histogram")
        pso_csr_hist, _ = dev.newComputePipelineStateWithFunction_error_(fn_csr_hist, None)
        
        fn_csr_pre = ecc_lib.newFunctionWithName_("msm_csr_prefix_sum")
        pso_csr_pre, _ = dev.newComputePipelineStateWithFunction_error_(fn_csr_pre, None)
        
        fn_csr_scat = ecc_lib.newFunctionWithName_("msm_csr_scatter")
        pso_csr_scat, _ = dev.newComputePipelineStateWithFunction_error_(fn_csr_scat, None)

        fn_msm1_v2 = ecc_lib.newFunctionWithName_("msm_bucket_accumulate_v2")
        if fn_msm1_v2 is None:
            raise RuntimeError("failed to find kernel function: msm_bucket_accumulate_v2")
        pso_msm1_v2, err = dev.newComputePipelineStateWithFunction_error_(fn_msm1_v2, None)
        if pso_msm1_v2 is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn_msm2_v2 = ecc_lib.newFunctionWithName_("msm_bucket_reduce_v2")
        if fn_msm2_v2 is None:
            raise RuntimeError("failed to find kernel function: msm_bucket_reduce_v2")
        pso_msm2_v2, err = dev.newComputePipelineStateWithFunction_error_(fn_msm2_v2, None)
        if pso_msm2_v2 is None:
            raise RuntimeError(f"failed to create pipeline state: {err}")

        fn_br = lib.newFunctionWithName_("ntt_bit_reverse")
        if fn_br is None:
            raise RuntimeError("failed to find kernel function: ntt_bit_reverse")
        pso_br, err = dev.newComputePipelineStateWithFunction_error_(fn_br, None)

        fn_bf = lib.newFunctionWithName_("ntt_butterfly")
        if fn_bf is None:
            raise RuntimeError("failed to find kernel function: ntt_butterfly")
        pso_bf, err = dev.newComputePipelineStateWithFunction_error_(fn_bf, None)

        fn_im = lib.newFunctionWithName_("intt_mul_inv_n")
        if fn_im is None:
            raise RuntimeError("failed to find kernel function: intt_mul_inv_n")
        pso_im, err = dev.newComputePipelineStateWithFunction_error_(fn_im, None)

        fn_st = lib.newFunctionWithName_("ntt_stockham")
        if fn_st is None:
            raise RuntimeError("failed to find kernel function: ntt_stockham")
        pso_st, err = dev.newComputePipelineStateWithFunction_error_(fn_st, None)

        q = dev.newCommandQueue()
        if q is None:
            raise RuntimeError("failed to create command queue")

        return MetalRuntime(
            device=dev,
            queue=q,
            lib=lib,
            ecc_lib=ecc_lib,
            pso_pointwise_mul_fr_mont=pso0,
            pso_poly_sub_fr_mont=pso_sub,
            pso_poly_scale_shift_fr_mont=pso_scale,
            pso_roots_evals_from_coeffs_fr_mont=pso1,
            pso_roots_coeffs_from_evals_fr_mont=pso2,
            pso_msm_bucket_accumulate=pso_msm1,
            pso_msm_bucket_reduce=pso_msm2,
            pso_msm_csr_histogram=pso_csr_hist,
            pso_msm_csr_prefix_sum=pso_csr_pre,
            pso_msm_csr_scatter=pso_csr_scat,
            pso_msm_bucket_accumulate_v2=pso_msm1_v2,
            pso_msm_bucket_reduce_v2=pso_msm2_v2,
            pso_ntt_bit_reverse=pso_br,
            pso_ntt_butterfly=pso_bf,
            pso_intt_mul_inv_n=pso_im,
            pso_ntt_stockham=pso_st,
        )
