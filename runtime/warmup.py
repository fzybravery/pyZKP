from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from pyZKP.backend.schemes.groth16.types import ProvingKey as Groth16ProvingKey
from pyZKP.backend.schemes.plonk.types import ProvingKey as PlonkProvingKey
from pyZKP.common.crypto.ecc.bn254 import G1
from pyZKP.common.crypto.msm import fixed_base_get_cached, fixed_base_precompute

# 缓存 points tuple
_POINTS_TUPLE_CACHE: Dict[int, Tuple[G1, ...]] = {}
# 记录同一个 Groth16 PK 被调用的次数
_GROTH16_PK_CALLS: Dict[int, int] = {} 


# 让 points 的tuple对象稳定，从而 fixed_base 的缓存能够命中
def cached_points_tuple(points: Sequence[G1]) -> Tuple[G1, ...]:
    if isinstance(points, tuple):
        return points
    key = id(points)
    out = _POINTS_TUPLE_CACHE.get(key)
    if out is not None:
        return out
    out = tuple(points)
    _POINTS_TUPLE_CACHE[key] = out
    return out


# 所有策略最终触发预热的底层函数
def warmup_fixed_base_points(points: Tuple[G1, ...], window_bits: int = 8) -> bool:
    if fixed_base_get_cached(points, window_bits) is not None:
        return False
    fixed_base_precompute(points, window_bits)
    return True

# 预热 Plonk 中的 fixed_base
def warmup_plonk_fixed_base(pk: PlonkProvingKey, *, n_points: int = 0, window_bits: int = 8) -> Dict[str, Any]:
    from pyZKP.runtime.kernels.cpu import kernels as cpu_kernels

    if n_points and int(n_points) > 0:
        n = int(n_points)
    else:
        n = int(
            max(
                pk.circuit.domain.n,
                len(pk.coeff_sigma1),
                len(pk.coeff_sigma2),
                len(pk.coeff_sigma3),
                len(pk.coeff_ql),
                len(pk.coeff_qr),
                len(pk.coeff_qm),
                len(pk.coeff_qo),
                len(pk.coeff_qc),
            )
        )
    points = cpu_kernels._srs_g1_prefix(pk.srs, n)
    did = warmup_fixed_base_points(points, window_bits=int(window_bits))
    return {"scheme": "plonk", "points_n": n, "window_bits": int(window_bits), "did_precompute": bool(did)}


# 预热 Groth16 中的 fixed_base
def warmup_groth16_fixed_base(pk: Groth16ProvingKey, *, window_bits: int = 8) -> Dict[str, Any]:
    w = int(window_bits)
    res = {"scheme": "groth16", "window_bits": w, "did_precompute": {}}

    a = cached_points_tuple(pk.a_query)
    res["did_precompute"]["a_query"] = warmup_fixed_base_points(a, window_bits=w)

    b1 = cached_points_tuple(pk.b_query_g1)
    res["did_precompute"]["b_query_g1"] = warmup_fixed_base_points(b1, window_bits=w)

    h = cached_points_tuple(pk.h_query)
    res["did_precompute"]["h_query"] = warmup_fixed_base_points(h, window_bits=w)

    if len(pk.l_query) != 0:
        l = cached_points_tuple(pk.l_query)
        res["did_precompute"]["l_query"] = warmup_fixed_base_points(l, window_bits=w)
    else:
        res["did_precompute"]["l_query"] = False

    return res

# 应用 fixed_base 策略
def apply_fixed_base_policy_plonk(pk: PlonkProvingKey, attrs: Dict[str, Any]) -> Dict[str, Any]:
    policy = str(attrs.get("fixed_base_policy", "off"))
    threshold = int(attrs.get("pippenger_threshold", 64))
    wbits = int(attrs.get("fixed_base_window_bits", attrs.get("warmup_fixed_base_window_bits", 8)))
    min_points = int(attrs.get("fixed_base_auto_min_points", 256))

    if policy == "on":
        attrs["fixed_base"] = True
        attrs["warmup_fixed_base"] = True
        attrs["warmup_fixed_base_window_bits"] = wbits

    if policy == "auto":
        n = int(
            max(
                pk.circuit.domain.n,
                len(pk.coeff_sigma1),
                len(pk.coeff_sigma2),
                len(pk.coeff_sigma3),
                len(pk.coeff_ql),
                len(pk.coeff_qr),
                len(pk.coeff_qm),
                len(pk.coeff_qo),
                len(pk.coeff_qc),
            )
        )
        if n >= threshold and n >= min_points:
            attrs["fixed_base"] = True
            attrs["warmup_fixed_base"] = True
            attrs["warmup_fixed_base_window_bits"] = wbits
            attrs["warmup_fixed_base_n"] = n
            attrs["fixed_base_auto_enabled"] = True

    if attrs.get("warmup_fixed_base"):
        warmup_plonk_fixed_base(
            pk,
            n_points=int(attrs.get("warmup_fixed_base_n", 0) or 0),
            window_bits=int(attrs.get("warmup_fixed_base_window_bits", 8) or 8),
        )
    return attrs

# 应用 fixed_base 策略
def apply_fixed_base_policy_groth16(pk: Groth16ProvingKey, attrs: Dict[str, Any]) -> Dict[str, Any]:
    policy = str(attrs.get("fixed_base_policy", "off"))
    threshold = int(attrs.get("pippenger_threshold", 64))
    wbits = int(attrs.get("fixed_base_window_bits", attrs.get("warmup_fixed_base_window_bits", 8)))
    min_points = int(attrs.get("fixed_base_auto_min_points", 256))
    max_len = max(len(pk.a_query), len(pk.b_query_g1), len(pk.h_query), len(pk.l_query))

    if policy == "on":
        attrs["fixed_base"] = True
        attrs["warmup_fixed_base"] = True
        attrs["warmup_fixed_base_window_bits"] = wbits

    if policy == "auto":
        k = id(pk)
        c = _GROTH16_PK_CALLS.get(k, 0) + 1
        _GROTH16_PK_CALLS[k] = c
        min_calls = int(attrs.get("fixed_base_auto_groth16_min_calls", 2))
        if c >= min_calls and max_len >= threshold and max_len >= min_points:
            attrs["fixed_base"] = True
            attrs["warmup_fixed_base"] = True
            attrs["warmup_fixed_base_window_bits"] = wbits
            attrs["fixed_base_auto_enabled"] = True

    if attrs.get("warmup_fixed_base"):
        warmup_groth16_fixed_base(pk, window_bits=int(attrs.get("warmup_fixed_base_window_bits", 8) or 8))
    return attrs
