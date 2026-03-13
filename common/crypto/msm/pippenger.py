from __future__ import annotations

from typing import Sequence

from pyZKP.common.crypto.ecc.bn254 import G1, G1_ZERO, g1_add, g1_mul
from pyZKP.common.crypto.field.fr import FR_MODULUS


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

    for k in range(0, max_bits, w):
        buckets = [G1_ZERO] * buckets_n
        for p, s in zip(points, scalars):
            ss = int(s) % FR_MODULUS
            digit = (ss >> k) & (buckets_n - 1)
            if digit == 0:
                continue
            buckets[digit] = g1_add(buckets[digit], p)

        running: G1 = G1_ZERO
        window_sum: G1 = G1_ZERO
        for digit in range(buckets_n - 1, 0, -1):
            running = g1_add(running, buckets[digit])
            window_sum = g1_add(window_sum, running)

        if k != 0:
            acc = g1_mul(acc, 1 << w)
        acc = g1_add(acc, window_sum)

    return acc
