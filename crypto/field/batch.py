from __future__ import annotations

from typing import List, Sequence

from crypto.field.fr import FR_MODULUS, fr_inv

# 使用蒙哥马利变换批量求逆元
def fr_batch_inv(xs: Sequence[int]) -> List[int]:
    n = len(xs)
    if n == 0:
        return []

    out = [0] * n
    prefix = [1] * (n + 1)
    nonzero = [False] * n

    acc = 1
    prefix[0] = 1
    for i, x in enumerate(xs):
        xi = int(x) % FR_MODULUS
        nonzero[i] = xi != 0
        if nonzero[i]:
            acc = (acc * xi) % FR_MODULUS
        prefix[i + 1] = acc

    if acc == 0:
        return [0] * n

    inv_acc = fr_inv(acc)
    suffix = inv_acc
    for i in range(n - 1, -1, -1):
        xi = int(xs[i]) % FR_MODULUS
        if nonzero[i]:
            out[i] = (prefix[i] * suffix) % FR_MODULUS
            suffix = (suffix * xi) % FR_MODULUS
        else:
            out[i] = 0
    return out

