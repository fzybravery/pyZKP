from __future__ import annotations

from typing import List, Sequence, Tuple

from common.crypto.field.fr import FR_MODULUS

# 对多项式进行除法，除数为 x^n - 1
def poly_div_by_xn_minus_1(num: Sequence[int], n: int) -> Tuple[List[int], List[int]]:
    if n <= 0:
        raise ValueError("n must be positive")
    a = [int(c) % FR_MODULUS for c in num]
    while len(a) > 0 and a[-1] % FR_MODULUS == 0:
        a.pop()
    if len(a) == 0:
        return [], []
    if len(a) <= n:
        return [], a

    m = len(a) - 1
    q_deg = m - n
    q = [0] * (q_deg + 1)

    for k in range(m, n - 1, -1):
        qk = q[k] if k <= q_deg else 0
        q[k - n] = (a[k] + qk) % FR_MODULUS

    r = [0] * n
    for i in range(n):
        qi = q[i] if i <= q_deg else 0
        ai = a[i] if i < len(a) else 0
        r[i] = (ai + qi) % FR_MODULUS

    while len(q) > 0 and q[-1] % FR_MODULUS == 0:
        q.pop()
    while len(r) > 0 and r[-1] % FR_MODULUS == 0:
        r.pop()
    return q, r
