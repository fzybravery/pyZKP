_ECC_KERNEL_SOURCE = r"""
// -----------------------------------------------------------------------------
// BN254 Fq 域 (基域) 操作
// -----------------------------------------------------------------------------

struct alignas(32) Fq {
    ulong v0;
    ulong v1;
    ulong v2;
    ulong v3;
};

// BN254 Base Field Modulus (Fq)
constant ulong FQ_MOD0 = 4332616871279656263ul;
constant ulong FQ_MOD1 = 10917124144477883021ul;
constant ulong FQ_MOD2 = 13281191951274694749ul;
constant ulong FQ_MOD3 = 3486998266802970665ul;

// Montgomery R mod Fq
constant ulong FQ_R_MOD0 = 15230403791020821917ul;
constant ulong FQ_R_MOD1 = 754611498739239741ul;
constant ulong FQ_R_MOD2 = 7381016538464732716ul;
constant ulong FQ_R_MOD3 = 1011752739694698287ul;

constant ulong FQ_INV_NEG = 9786893198990664585ul;

inline bool geq_mod_fq(Fq a) {
    if (a.v3 != FQ_MOD3) return a.v3 > FQ_MOD3;
    if (a.v2 != FQ_MOD2) return a.v2 > FQ_MOD2;
    if (a.v1 != FQ_MOD1) return a.v1 > FQ_MOD1;
    return a.v0 >= FQ_MOD0;
}

inline Fq sub_mod_fq(Fq a) {
    ulong borrow = 0;
    a.v0 = subb64(a.v0, FQ_MOD0, borrow);
    a.v1 = subb64(a.v1, FQ_MOD1, borrow);
    a.v2 = subb64(a.v2, FQ_MOD2, borrow);
    a.v3 = subb64(a.v3, FQ_MOD3, borrow);
    return a;
}

inline Fq add_mod_fq(Fq a, Fq b) {
    ulong carry = 0;
    Fq r;
    r.v0 = addc64(a.v0, b.v0, carry);
    r.v1 = addc64(a.v1, b.v1, carry);
    r.v2 = addc64(a.v2, b.v2, carry);
    r.v3 = addc64(a.v3, b.v3, carry);
    if (carry != 0 || geq_mod_fq(r)) {
        r = sub_mod_fq(r);
    }
    return r;
}

inline Fq sub_mod2_fq(Fq a, Fq b) {
    ulong borrow = 0;
    Fq r;
    r.v0 = subb64(a.v0, b.v0, borrow);
    r.v1 = subb64(a.v1, b.v1, borrow);
    r.v2 = subb64(a.v2, b.v2, borrow);
    r.v3 = subb64(a.v3, b.v3, borrow);
    if (borrow != 0) {
        ulong carry = 0;
        r.v0 = addc64(r.v0, FQ_MOD0, carry);
        r.v1 = addc64(r.v1, FQ_MOD1, carry);
        r.v2 = addc64(r.v2, FQ_MOD2, carry);
        r.v3 = addc64(r.v3, FQ_MOD3, carry);
    }
    return r;
}

inline Fq mont_mul_fq(Fq a, Fq b) {
    ulong t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

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
        ulong m = t[i] * FQ_INV_NEG;

        ulong carry = 0;
        mac64(m, FQ_MOD0, t[i+0], carry);
        mac64(m, FQ_MOD1, t[i+1], carry);
        mac64(m, FQ_MOD2, t[i+2], carry);
        mac64(m, FQ_MOD3, t[i+3], carry);

        ulong c = 0;
        t[i+4] = addc64(t[i+4], carry, c);
        uint k = i + 5;
        while (c != 0 && k < 8) {
            t[k] = addc64(t[k], 0ul, c);
            k++;
        }
    }

    Fq r;
    r.v0 = t[4];
    r.v1 = t[5];
    r.v2 = t[6];
    r.v3 = t[7];
    if (geq_mod_fq(r)) {
        r = sub_mod_fq(r);
    }
    return r;
}

// -----------------------------------------------------------------------------
// G1 点加法 (Jacobian 坐标系)
// -----------------------------------------------------------------------------

struct alignas(32) G1Point {
    Fq x;
    Fq y;
    Fq z;
};

inline bool is_zero_g1(G1Point p) {
    return (p.z.v0 == 0 && p.z.v1 == 0 && p.z.v2 == 0 && p.z.v3 == 0);
}

// Jacobian 点倍加: R = 2 * P
inline G1Point g1_double_jac(G1Point p) {
    if (is_zero_g1(p)) return p;
    
    // A = X1^2
    Fq A = mont_mul_fq(p.x, p.x);
    // B = Y1^2
    Fq B = mont_mul_fq(p.y, p.y);
    // C = B^2
    Fq C = mont_mul_fq(B, B);
    
    // X1_plus_B = X1 + B
    Fq X1_plus_B = add_mod_fq(p.x, B);
    // X1_plus_B_sq = (X1 + B)^2
    Fq X1_plus_B_sq = mont_mul_fq(X1_plus_B, X1_plus_B);
    
    // D_1 = (X1 + B)^2 - A - C
    Fq D_1 = sub_mod2_fq(X1_plus_B_sq, A);
    D_1 = sub_mod2_fq(D_1, C);
    
    // D = 2 * D_1
    Fq D = add_mod_fq(D_1, D_1);
    
    // E = 3 * A
    Fq E = add_mod_fq(A, add_mod_fq(A, A));
    // F = E^2
    Fq F = mont_mul_fq(E, E);
    
    // X3 = F - 2 * D
    Fq D_2 = add_mod_fq(D, D);
    Fq X3 = sub_mod2_fq(F, D_2);
    
    // Y3 = E * (D - X3) - 8 * C
    Fq D_minus_X3 = sub_mod2_fq(D, X3);
    Fq Y3_1 = mont_mul_fq(E, D_minus_X3);
    
    Fq C_8 = add_mod_fq(C, C);
    C_8 = add_mod_fq(C_8, C_8);
    C_8 = add_mod_fq(C_8, C_8);
    
    Fq Y3 = sub_mod2_fq(Y3_1, C_8);
    
    // Z3 = 2 * Y1 * Z1
    Fq Y1_Z1 = mont_mul_fq(p.y, p.z);
    Fq Z3 = add_mod_fq(Y1_Z1, Y1_Z1);
    
    G1Point res;
    res.x = X3;
    res.y = Y3;
    res.z = Z3;
    return res;
}

// 完整的 Jacobian 点加法: R = P + Q
inline G1Point g1_add_jac(G1Point p, G1Point q) {
    if (is_zero_g1(p)) return q;
    if (is_zero_g1(q)) return p;

    // Z1Z1 = Z1^2
    Fq Z1Z1 = mont_mul_fq(p.z, p.z);
    // Z2Z2 = Z2^2
    Fq Z2Z2 = mont_mul_fq(q.z, q.z);
    
    // U1 = X1 * Z2Z2
    Fq U1 = mont_mul_fq(p.x, Z2Z2);
    // U2 = X2 * Z1Z1
    Fq U2 = mont_mul_fq(q.x, Z1Z1);
    
    // S1 = Y1 * Z2 * Z2Z2
    Fq Z2_Z2Z2 = mont_mul_fq(q.z, Z2Z2);
    Fq S1 = mont_mul_fq(p.y, Z2_Z2Z2);
    
    // S2 = Y2 * Z1 * Z1Z1
    Fq Z1_Z1Z1 = mont_mul_fq(p.z, Z1Z1);
    Fq S2 = mont_mul_fq(q.y, Z1_Z1Z1);

    // H = U2 - U1
    Fq H = sub_mod2_fq(U2, U1);
    
    bool H_is_zero = (H.v0 == 0 && H.v1 == 0 && H.v2 == 0 && H.v3 == 0);
    
    // r = S2 - S1
    Fq R = sub_mod2_fq(S2, S1);
    
    if (H_is_zero) {
        bool R_is_zero = (R.v0 == 0 && R.v1 == 0 && R.v2 == 0 && R.v3 == 0);
        if (R_is_zero) {
            // P == Q, 调用倍加
            return g1_double_jac(p);
        } else {
            // P == -Q
            G1Point res;
            res.x.v0 = 0; res.x.v1 = 0; res.x.v2 = 0; res.x.v3 = 0;
            res.y.v0 = 0; res.y.v1 = 0; res.y.v2 = 0; res.y.v3 = 0;
            res.z.v0 = 0; res.z.v1 = 0; res.z.v2 = 0; res.z.v3 = 0;
            return res;
        }
    }

    // HH = H^2
    Fq HH = mont_mul_fq(H, H);
    // I = 4 * HH
    Fq I = add_mod_fq(HH, HH);
    I = add_mod_fq(I, I);
    
    // J = H * I
    Fq J = mont_mul_fq(H, I);
    
    // r = 2 * r
    Fq R2 = add_mod_fq(R, R);
    
    // V = U1 * I
    Fq V = mont_mul_fq(U1, I);
    
    // X3 = r^2 - J - 2*V
    Fq X3 = mont_mul_fq(R2, R2);
    X3 = sub_mod2_fq(X3, J);
    X3 = sub_mod2_fq(X3, V);
    X3 = sub_mod2_fq(X3, V);
    
    // Y3 = r * (V - X3) - 2 * S1 * J
    Fq Y3_1 = sub_mod2_fq(V, X3);
    Y3_1 = mont_mul_fq(R2, Y3_1);
    
    Fq S1_J = mont_mul_fq(S1, J);
    Fq S1_J_2 = add_mod_fq(S1_J, S1_J);
    
    Fq Y3 = sub_mod2_fq(Y3_1, S1_J_2);
    
    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    Fq Z1_plus_Z2 = add_mod_fq(p.z, q.z);
    Fq Z3_1 = mont_mul_fq(Z1_plus_Z2, Z1_plus_Z2);
    Z3_1 = sub_mod2_fq(Z3_1, Z1Z1);
    Z3_1 = sub_mod2_fq(Z3_1, Z2Z2);
    Fq Z3 = mont_mul_fq(Z3_1, H);
    
    G1Point res;
    res.x = X3;
    res.y = Y3;
    res.z = Z3;
    return res;
}

// -----------------------------------------------------------------------------
// Pippenger MSM Phase 1: Bucket 累加
// -----------------------------------------------------------------------------

// points: [N] 预先上传到显存的基点
// scalars: [N] 预先上传到显存的标量 (Fr)
// buckets: [window_count * (1<<window_bits)] 所有的桶，按 (window_idx, bucket_idx) 平铺
// 每个线程处理一个标量，将其累加到对应的 bucket 中。
// 注意：由于存在读写冲突（多个标量可能被分配到同一个 bucket），需要使用原子操作或者按照 bucket_idx 归约。
// 为了在 GPU 上简单实现，Phase1 我们采用以 Bucket 为中心的并行：
// 每个线程负责一个 Bucket，遍历所有标量，提取属于自己的部分进行点加。

kernel void msm_bucket_accumulate(
    const device G1Point* points [[buffer(0)]],
    const device Fr* scalars [[buffer(1)]],
    device G1Point* buckets [[buffer(2)]],
    constant uint& num_points [[buffer(3)]],
    constant uint& window_bits [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid 是 bucket 的全局 ID。
    // window_idx = gid >> window_bits
    // bucket_idx = gid & ((1 << window_bits) - 1)
    
    uint buckets_per_window = 1 << window_bits;
    uint window_idx = gid / buckets_per_window;
    uint bucket_idx = gid % buckets_per_window;
    
    // 0 号 bucket 不累加点
    if (bucket_idx == 0) return;
    
    uint bit_offset = window_idx * window_bits;
    
    G1Point acc;
    acc.x.v0 = 0; acc.x.v1 = 0; acc.x.v2 = 0; acc.x.v3 = 0;
    acc.y.v0 = 0; acc.y.v1 = 0; acc.y.v2 = 0; acc.y.v3 = 0;
    acc.z.v0 = 0; acc.z.v1 = 0; acc.z.v2 = 0; acc.z.v3 = 0;
    
    for (uint i = 0; i < num_points; i++) {
        Fr s_mont = scalars[i];
        
        // s_mont is in Montgomery form (s * R mod P). We need standard representation to extract bits.
        // Multiply by 1 (standard 1) in Montgomery domain: mont_mul(s_mont, 1) = s_mont * 1 * R^-1 = s.
        Fr std_one;
        std_one.v0 = 1; std_one.v1 = 0; std_one.v2 = 0; std_one.v3 = 0;
        Fr s = mont_mul(s_mont, std_one);
        
        // 提取第 bit_offset 到 bit_offset + window_bits - 1 位
        // 这里为了简化，我们假设 window_bits 最大不超过 32，且跨 64 位边界的处理
        uint word_idx = bit_offset / 64;
        uint bit_in_word = bit_offset % 64;
        
        ulong s_word = 0;
        ulong next_word = 0;
        if (word_idx == 0) { s_word = s.v0; next_word = s.v1; }
        else if (word_idx == 1) { s_word = s.v1; next_word = s.v2; }
        else if (word_idx == 2) { s_word = s.v2; next_word = s.v3; }
        else if (word_idx == 3) { s_word = s.v3; next_word = 0; }
        
        uint extracted = 0;
        if (bit_in_word + window_bits <= 64) {
            extracted = (s_word >> bit_in_word) & ((1ul << window_bits) - 1);
        } else {
            uint bits_in_first = 64 - bit_in_word;
            uint bits_in_second = window_bits - bits_in_first;
            uint mask1 = (1ul << bits_in_first) - 1;
            uint mask2 = (1ul << bits_in_second) - 1;
            
            uint part1 = (s_word >> bit_in_word) & mask1;
            uint part2 = (next_word & mask2) << bits_in_first;
            extracted = part1 | part2;
        }
        
        if (extracted == bucket_idx) {
            acc = g1_add_jac(acc, points[i]);
        }
    }
    
    buckets[gid] = acc;
}
// -----------------------------------------------------------------------------
// Pippenger MSM Phase 2: Window 内 Bucket 聚合
// -----------------------------------------------------------------------------
// 对于每一个 window，我们需要把它的所有 bucket 按照 Pippenger 算法聚合起来：
// running_sum = 0
// window_sum = 0
// for d = buckets_n - 1 down to 1:
//     running_sum = running_sum + buckets[d]
//     window_sum = window_sum + running_sum
// 最终该 window 的结果就是 window_sum。

kernel void msm_bucket_reduce(
    const device G1Point* buckets [[buffer(0)]],
    device G1Point* window_sums [[buffer(1)]],
    constant uint& window_bits [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid 代表当前处理的是第几个 window
    uint buckets_per_window = 1 << window_bits;
    uint offset = gid * buckets_per_window;
    
    G1Point running_sum;
    running_sum.x.v0 = 0; running_sum.x.v1 = 0; running_sum.x.v2 = 0; running_sum.x.v3 = 0;
    running_sum.y.v0 = 0; running_sum.y.v1 = 0; running_sum.y.v2 = 0; running_sum.y.v3 = 0;
    running_sum.z.v0 = 0; running_sum.z.v1 = 0; running_sum.z.v2 = 0; running_sum.z.v3 = 0;
    
    G1Point window_sum;
    window_sum.x.v0 = 0; window_sum.x.v1 = 0; window_sum.x.v2 = 0; window_sum.x.v3 = 0;
    window_sum.y.v0 = 0; window_sum.y.v1 = 0; window_sum.y.v2 = 0; window_sum.y.v3 = 0;
    window_sum.z.v0 = 0; window_sum.z.v1 = 0; window_sum.z.v2 = 0; window_sum.z.v3 = 0;
    
    // 倒序遍历 bucket
    for (int d = buckets_per_window - 1; d > 0; d--) {
        G1Point b = buckets[offset + d];
        running_sum = g1_add_jac(running_sum, b);
        window_sum = g1_add_jac(window_sum, running_sum);
    }
    
    window_sums[gid] = window_sum;
}
"""
