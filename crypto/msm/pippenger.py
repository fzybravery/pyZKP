from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from crypto.ecc.bn254 import G1, G1_ZERO, G2, G2_ZERO, g1_add, g1_mul, g2_add, g2_mul
from crypto.field.fr import FR_MODULUS

# Piggenger 多标量乘法的高效实现
def msm_pippenger(points: Sequence[G1], scalars: Sequence[int], window_bits: int = 16) -> G1:
    if len(points) != len(scalars):
        raise ValueError("points and scalars length mismatch")
    if len(points) == 0:
        return G1_ZERO
    w = int(window_bits)
    if w <= 0 or w > 20:
        raise ValueError("invalid window_bits")

    max_bits = FR_MODULUS.bit_length()
    buckets_n = 1 << w
    acc: G1 = G1_ZERO

    for k in reversed(range(0, max_bits, w)):
        window_sum: G1 = G1_ZERO
        if buckets_n <= 1024:
            buckets = [G1_ZERO] * buckets_n
            for p, s in zip(points, scalars):
                ss = int(s) % FR_MODULUS
                digit = (ss >> k) & (buckets_n - 1)
                if digit == 0:
                    continue
                buckets[digit] = g1_add(buckets[digit], p)

            running: G1 = G1_ZERO
            for digit in range(buckets_n - 1, 0, -1):
                running = g1_add(running, buckets[digit])
                window_sum = g1_add(window_sum, running)
        else:
            buckets2: Dict[int, G1] = {}
            for p, s in zip(points, scalars):
                ss = int(s) % FR_MODULUS
                digit = (ss >> k) & (buckets_n - 1)
                if digit == 0:
                    continue
                prev = buckets2.get(digit)
                buckets2[digit] = g1_add(prev, p) if prev is not None else p

            running = G1_ZERO
            prev_digit = buckets_n - 1
            for digit in sorted(buckets2.keys(), reverse=True):
                gap = prev_digit - digit - 1
                if gap > 0 and running != G1_ZERO:
                    window_sum = g1_add(window_sum, g1_mul(running, gap))
                running = g1_add(running, buckets2[digit])
                window_sum = g1_add(window_sum, running)
                prev_digit = digit
            tail = prev_digit - 1
            if tail > 0 and running != G1_ZERO:
                window_sum = g1_add(window_sum, g1_mul(running, tail))

        acc = g1_mul(acc, 1 << w)
        acc = g1_add(acc, window_sum)

    return acc


def msm_pippenger_g2(points: Sequence[G2], scalars: Sequence[int], window_bits: int = 16) -> G2:
    if len(points) != len(scalars):
        raise ValueError("points and scalars length mismatch")
    if len(points) == 0:
        return G2_ZERO
    w = int(window_bits)
    if w <= 0 or w > 20:
        raise ValueError("invalid window_bits")

    max_bits = FR_MODULUS.bit_length()
    buckets_n = 1 << w
    acc: G2 = G2_ZERO

    for k in reversed(range(0, max_bits, w)):
        window_sum: G2 = G2_ZERO
        if buckets_n <= 1024:
            buckets = [G2_ZERO] * buckets_n
            for p, s in zip(points, scalars):
                ss = int(s) % FR_MODULUS
                digit = (ss >> k) & (buckets_n - 1)
                if digit == 0:
                    continue
                buckets[digit] = g2_add(buckets[digit], p)

            running: G2 = G2_ZERO
            for digit in range(buckets_n - 1, 0, -1):
                running = g2_add(running, buckets[digit])
                window_sum = g2_add(window_sum, running)
        else:
            buckets2: Dict[int, G2] = {}
            for p, s in zip(points, scalars):
                ss = int(s) % FR_MODULUS
                digit = (ss >> k) & (buckets_n - 1)
                if digit == 0:
                    continue
                prev = buckets2.get(digit)
                buckets2[digit] = g2_add(prev, p) if prev is not None else p

            running = G2_ZERO
            prev_digit = buckets_n - 1
            for digit in sorted(buckets2.keys(), reverse=True):
                gap = prev_digit - digit - 1
                if gap > 0 and running != G2_ZERO:
                    window_sum = g2_add(window_sum, g2_mul(running, gap))
                running = g2_add(running, buckets2[digit])
                window_sum = g2_add(window_sum, running)
                prev_digit = digit
            tail = prev_digit - 1
            if tail > 0 and running != G2_ZERO:
                window_sum = g2_add(window_sum, g2_mul(running, tail))

        acc = g2_mul(acc, 1 << w)
        acc = g2_add(acc, window_sum)

    return acc


def msm_pippenger_batch(points: Sequence[G1], scalars_list: Sequence[Sequence[int]], window_bits: int = 16) -> List[G1]:
    m = len(scalars_list)
    if m == 0:
        return []
    w = int(window_bits)
    if w <= 0 or w > 20:
        raise ValueError("invalid window_bits")
    lens = [len(s) for s in scalars_list]
    max_len = max(lens) if lens else 0
    if max_len == 0:
        return [G1_ZERO] * m
    if len(points) < max_len:
        raise ValueError("points and scalars length mismatch")
    uniform = all(l == max_len for l in lens)

    max_bits = FR_MODULUS.bit_length()
    buckets_n = 1 << w
    accs: List[G1] = [G1_ZERO] * m
    pts = points[:max_len]

    for k in reversed(range(0, max_bits, w)):
        if buckets_n <= 1024:
            buckets = [[G1_ZERO] * buckets_n for _ in range(m)]
            if uniform:
                for i, p in enumerate(pts):
                    for j in range(m):
                        ss = int(scalars_list[j][i]) % FR_MODULUS
                        digit = (ss >> k) & (buckets_n - 1)
                        if digit == 0:
                            continue
                        buckets[j][digit] = g1_add(buckets[j][digit], p)
            else:
                for i, p in enumerate(pts):
                    for j in range(m):
                        if i >= lens[j]:
                            continue
                        ss = int(scalars_list[j][i]) % FR_MODULUS
                        digit = (ss >> k) & (buckets_n - 1)
                        if digit == 0:
                            continue
                        buckets[j][digit] = g1_add(buckets[j][digit], p)

            for j in range(m):
                running = G1_ZERO
                window_sum = G1_ZERO
                for digit in range(buckets_n - 1, 0, -1):
                    running = g1_add(running, buckets[j][digit])
                    window_sum = g1_add(window_sum, running)
                accs[j] = g1_mul(accs[j], 1 << w)
                accs[j] = g1_add(accs[j], window_sum)
        else:
            buckets2: List[Dict[int, G1]] = [{} for _ in range(m)]
            if uniform:
                for i, p in enumerate(pts):
                    for j in range(m):
                        ss = int(scalars_list[j][i]) % FR_MODULUS
                        digit = (ss >> k) & (buckets_n - 1)
                        if digit == 0:
                            continue
                        prev = buckets2[j].get(digit)
                        buckets2[j][digit] = g1_add(prev, p) if prev is not None else p
            else:
                for i, p in enumerate(pts):
                    for j in range(m):
                        if i >= lens[j]:
                            continue
                        ss = int(scalars_list[j][i]) % FR_MODULUS
                        digit = (ss >> k) & (buckets_n - 1)
                        if digit == 0:
                            continue
                        prev = buckets2[j].get(digit)
                        buckets2[j][digit] = g1_add(prev, p) if prev is not None else p

            for j in range(m):
                window_sum = G1_ZERO
                running = G1_ZERO
                prev_digit = buckets_n - 1
                for digit in sorted(buckets2[j].keys(), reverse=True):
                    gap = prev_digit - digit - 1
                    if gap > 0 and running != G1_ZERO:
                        window_sum = g1_add(window_sum, g1_mul(running, gap))
                    running = g1_add(running, buckets2[j][digit])
                    window_sum = g1_add(window_sum, running)
                    prev_digit = digit
                tail = prev_digit - 1
                if tail > 0 and running != G1_ZERO:
                    window_sum = g1_add(window_sum, g1_mul(running, tail))
                accs[j] = g1_mul(accs[j], 1 << w)
                accs[j] = g1_add(accs[j], window_sum)

    return accs


@dataclass(frozen=True)
class FixedBasePrecomp:
    points: Tuple[G1, ...] # 原始基点
    window_bits: int # 切片窗口大小
    table: Tuple[Tuple[G1, ...], ...] # 核心预计算表


# 全局缓存，确保同一个基点只需要计算一次
_FIXED_BASE_CACHE: Dict[Tuple[int, int], FixedBasePrecomp] = {} # 乘法表
_FIXED_BASE_USE_COUNT: Dict[Tuple[int, int], int] = {} # 存储某组基点被查询的次数

# 缓存查询接口
def fixed_base_get_cached(points: Tuple[G1, ...], window_bits: int) -> Optional[FixedBasePrecomp]:
    w = int(window_bits)
    return _FIXED_BASE_CACHE.get((id(points), w))


def fixed_base_put_cached(precomp: FixedBasePrecomp) -> None:
    _FIXED_BASE_CACHE[(id(precomp.points), int(precomp.window_bits))] = precomp

# 智能按需预计算
def fixed_base_maybe_precompute(points: Tuple[G1, ...], window_bits: int, min_uses: int = 2) -> Optional[FixedBasePrecomp]:
    w = int(window_bits)
    key = (id(points), w)
    cached = _FIXED_BASE_CACHE.get(key)
    if cached is not None:
        return cached
    c = _FIXED_BASE_USE_COUNT.get(key, 0) + 1
    _FIXED_BASE_USE_COUNT[key] = c
    if c < int(min_uses):
        return None
    return fixed_base_precompute(points, w)


# 预计算函数
def fixed_base_precompute(points: Sequence[G1], window_bits: int) -> FixedBasePrecomp:
    w = int(window_bits)
    if w <= 0 or w > 12:
        raise ValueError("invalid window_bits")
    pts = points if isinstance(points, tuple) else tuple(points)
    key = (id(pts), w)
    cached = _FIXED_BASE_CACHE.get(key)
    if cached is not None:
        return cached
    buckets_n = 1 << w
    table: List[Tuple[G1, ...]] = []
    for p in pts:
        row = [G1_ZERO] * buckets_n
        acc = G1_ZERO
        for d in range(1, buckets_n):
            acc = g1_add(acc, p)
            row[d] = acc
        table.append(tuple(row))
    out = FixedBasePrecomp(points=pts, window_bits=w, table=tuple(table))
    _FIXED_BASE_CACHE[key] = out
    return out


# 单次固定基点msm
def msm_fixed_base(precomp: FixedBasePrecomp, scalars: Sequence[int]) -> G1:
    if len(precomp.points) != len(scalars):
        raise ValueError("points and scalars length mismatch")
    if len(scalars) == 0:
        return G1_ZERO
    w = int(precomp.window_bits)
    max_bits = FR_MODULUS.bit_length()
    mask = (1 << w) - 1
    acc: G1 = G1_ZERO
    for k in reversed(range(0, max_bits, w)):
        window_sum: G1 = G1_ZERO
        for i, s in enumerate(scalars):
            ss = int(s) % FR_MODULUS
            digit = (ss >> k) & mask
            if digit == 0:
                continue
            window_sum = g1_add(window_sum, precomp.table[i][digit])
        acc = g1_mul(acc, 1 << w)
        acc = g1_add(acc, window_sum)
    return acc

# 批处理固定基点msm，提高cache命中率
def msm_fixed_base_batch(precomp: FixedBasePrecomp, scalars_list: Sequence[Sequence[int]]) -> List[G1]:
    m = len(scalars_list)
    if m == 0:
        return []
    lens = [len(s) for s in scalars_list]
    max_len = max(lens) if lens else 0
    if max_len == 0:
        return [G1_ZERO] * m
    if len(precomp.points) < max_len:
        raise ValueError("points and scalars length mismatch")
    if precomp.window_bits <= 0:
        raise ValueError("invalid window_bits")
    w = int(precomp.window_bits)
    max_bits = FR_MODULUS.bit_length()
    mask = (1 << w) - 1
    uniform = all(l == max_len for l in lens)

    accs: List[G1] = [G1_ZERO] * m
    for k in reversed(range(0, max_bits, w)):
        window_sums: List[G1] = [G1_ZERO] * m
        if uniform:
            for i in range(max_len):
                row = precomp.table[i]
                for j in range(m):
                    ss = int(scalars_list[j][i]) % FR_MODULUS
                    digit = (ss >> k) & mask
                    if digit == 0:
                        continue
                    window_sums[j] = g1_add(window_sums[j], row[digit])
        else:
            for i in range(max_len):
                row = precomp.table[i]
                for j in range(m):
                    if i >= lens[j]:
                        continue
                    ss = int(scalars_list[j][i]) % FR_MODULUS
                    digit = (ss >> k) & mask
                    if digit == 0:
                        continue
                    window_sums[j] = g1_add(window_sums[j], row[digit])

        for j in range(m):
            accs[j] = g1_mul(accs[j], 1 << w)
            accs[j] = g1_add(accs[j], window_sums[j])
    return accs
