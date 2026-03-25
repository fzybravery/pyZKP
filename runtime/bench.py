from __future__ import annotations

"""
runtime bench（最小可复现基准）。

输出 JSON，包含：
- prove/verify 总耗时（s）
- runtime trace：按 op 聚合的耗时占比
- 内存池统计：分配次数/复用次数/峰值 in-use

注意：
- 该 bench 仅用于建立 CPU baseline 与 trace 口径，后续接入 GPU 后可复用同一输出格式做对比。
"""

import argparse
import json
import os
import pickle
import time

from pyZKP import build_witness, check_r1cs, compile_circuit
from pyZKP.backend.schemes.groth16.prove import prove as groth16_prove
from pyZKP.backend.schemes.groth16.setup import setup as groth16_setup
from pyZKP.backend.schemes.groth16.verify import verify as groth16_verify
from pyZKP.backend.schemes.plonk import setup as plonk_setup
from pyZKP.backend.schemes.plonk.prove import prove as plonk_prove
from pyZKP.backend.schemes.plonk.verify import verify as plonk_verify
from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.msm import fixed_base_get_cached, fixed_base_precompute
from pyZKP.frontend.circuit.schema import public, secret
from pyZKP.runtime.memory import CPUMemoryPool
from pyZKP.runtime.trace import Trace


class RepeatMulCircuit:
    """
    简单基准电路：
    - repeat 次重复乘法，扩大约束规模
    - 最后约束 y == x^3 + x + 5
    """
    def __init__(self, repeat: int) -> None:
        self.repeat = int(repeat)
        self.x = secret("x")
        self.y = public("y")

    def define(self, api) -> None:
        for _ in range(self.repeat):
            api.Mul(self.x, self.x)
        x3 = api.Mul(self.x, self.x, self.x)
        api.AssertIsEqual(self.y, api.Add(x3, self.x, 5))


def _ns() -> int:
    return time.perf_counter_ns()

def _s(ns: int) -> float:
    return float(ns) / 1_000_000_000.0


def _default_setup_cache_path(*, scheme: str, repeat: int) -> str:
    return os.path.join(".pyZKP_bench_cache", f"{scheme}_repeat{int(repeat)}_mod{int(FR_MODULUS)}_v1.pkl")


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: str, obj) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def bench_plonk(
    *,
    repeat: int,
    batch: int,
    setup_cache_path: str | None = None,
    no_setup_cache: bool = False,
    warmup_fixed_base: bool = False,
    warmup_fixed_base_window_bits: int = 8,
    warmup_fixed_base_n: int = 0,
    fixed_base_policy: str = "off",
    fixed_base_window_bits: int = 8,
    fixed_base_auto_min_points: int = 256,
) -> dict:
    """
    运行 PLONK 的 setup/prove/verify，并返回性能统计。
    """
    total0 = _ns()
    c0 = _ns()
    ir = compile_circuit(RepeatMulCircuit(repeat), FR_MODULUS)
    c1 = _ns()
    x = 9
    y = (x * x * x + x + 5) % FR_MODULUS
    w0 = _ns()
    wit = build_witness(ir, {"x": x, "y": y})
    w1 = _ns()
    r0 = _ns()
    check_r1cs(ir, wit)
    r1 = _ns()

    setup_cache_path = setup_cache_path or _default_setup_cache_path(scheme="plonk", repeat=repeat)
    setup_cache_hit = False
    setup_load_s = 0.0
    setup_save_s = 0.0
    setup_compute_s = 0.0

    s0 = _ns()
    if (not no_setup_cache) and os.path.exists(setup_cache_path):
        l0 = _ns()
        pk = _load_pickle(setup_cache_path)
        l1 = _ns()
        setup_cache_hit = True
        setup_load_s = _s(l1 - l0)
    else:
        sc0 = _ns()
        pk = plonk_setup(ir)
        sc1 = _ns()
        setup_compute_s = _s(sc1 - sc0)
        if not no_setup_cache:
            sv0 = _ns()
            _save_pickle(setup_cache_path, pk)
            sv1 = _ns()
            setup_save_s = _s(sv1 - sv0)
    s1 = _ns()
    public_values = [1, y]

    warmup_fixed_base_s = 0.0
    warmup_fixed_base_cached = False
    warmup_fixed_base_points_n = 0
    if warmup_fixed_base:
        from pyZKP.runtime.kernels.cpu import kernels as cpu_kernels

        if warmup_fixed_base_n and int(warmup_fixed_base_n) > 0:
            n_points = int(warmup_fixed_base_n)
        else:
            n_points = int(
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
        points = cpu_kernels._srs_g1_prefix(pk.srs, n_points)
        warmup_fixed_base_points_n = n_points
        cached = fixed_base_get_cached(points, warmup_fixed_base_window_bits)
        if cached is not None:
            warmup_fixed_base_cached = True
        else:
            wb0 = _ns()
            fixed_base_precompute(points, warmup_fixed_base_window_bits)
            wb1 = _ns()
            warmup_fixed_base_s = _s(wb1 - wb0)

    pool = CPUMemoryPool()
    trace = Trace()
    runtime_attrs = {
        "fixed_base_policy": str(fixed_base_policy),
        "fixed_base_window_bits": int(fixed_base_window_bits),
        "fixed_base_auto_min_points": int(fixed_base_auto_min_points),
    }

    t0 = _ns()
    proofs = []
    for _ in range(batch):
        proofs.append(plonk_prove(pk, wit, public_values=public_values, runtime_trace=trace, runtime_pool=pool, runtime_attrs=runtime_attrs))
    t1 = _ns()

    v0 = _ns()
    oks = [plonk_verify(pk.vk, p, public_values=public_values) for p in proofs]
    v1 = _ns()

    if not all(oks):
        raise RuntimeError("plonk verify failed")
    total1 = _ns()

    return {
        "scheme": "plonk",
        "repeat": repeat,
        "batch": batch,
        "total_s": _s(total1 - total0),
        "compile_s": _s(c1 - c0),
        "witness_s": _s(w1 - w0),
        "check_r1cs_s": _s(r1 - r0),
        "setup_s": _s(s1 - s0),
        "setup_cache_path": setup_cache_path,
        "setup_cache_hit": setup_cache_hit,
        "setup_load_s": setup_load_s,
        "setup_compute_s": setup_compute_s,
        "setup_save_s": setup_save_s,
        "warmup_fixed_base": bool(warmup_fixed_base),
        "warmup_fixed_base_window_bits": int(warmup_fixed_base_window_bits),
        "warmup_fixed_base_points_n": int(warmup_fixed_base_points_n),
        "warmup_fixed_base_cached": bool(warmup_fixed_base_cached),
        "warmup_fixed_base_s": float(warmup_fixed_base_s),
        "fixed_base_policy": str(fixed_base_policy),
        "fixed_base_window_bits": int(fixed_base_window_bits),
        "fixed_base_auto_min_points": int(fixed_base_auto_min_points),
        "prove_s": _s(t1 - t0),
        "verify_s": _s(v1 - v0),
        "trace_total_s": _s(trace.total_ns()),
        "trace_by_op_s": {k: _s(v) for k, v in trace.summarize_ns_by_op().items()},
        "pool": {
            "alloc_calls": pool.stats.alloc_calls,
            "reuse_calls": pool.stats.reuse_calls,
            "peak_in_use": pool.stats.peak_in_use,
        },
    }


def bench_groth16(
    *,
    repeat: int,
    batch: int,
    setup_cache_path: str | None = None,
    no_setup_cache: bool = False,
    fixed_base_policy: str = "off",
    fixed_base_window_bits: int = 8,
    fixed_base_auto_min_points: int = 256,
    fixed_base_auto_groth16_min_calls: int = 2,
) -> dict:
    """
    运行 Groth16 的 setup/prove/verify，并返回性能统计。
    """
    total0 = _ns()
    c0 = _ns()
    ir = compile_circuit(RepeatMulCircuit(repeat), FR_MODULUS)
    c1 = _ns()
    x = 9
    y = (x * x * x + x + 5) % FR_MODULUS
    w0 = _ns()
    wit = build_witness(ir, {"x": x, "y": y})
    w1 = _ns()
    r0 = _ns()
    check_r1cs(ir, wit)
    r1 = _ns()

    setup_cache_path = setup_cache_path or _default_setup_cache_path(scheme="groth16", repeat=repeat)
    setup_cache_hit = False
    setup_load_s = 0.0
    setup_save_s = 0.0
    setup_compute_s = 0.0

    s0 = _ns()
    if (not no_setup_cache) and os.path.exists(setup_cache_path):
        l0 = _ns()
        pk = _load_pickle(setup_cache_path)
        l1 = _ns()
        setup_cache_hit = True
        setup_load_s = _s(l1 - l0)
    else:
        sc0 = _ns()
        pk = groth16_setup(ir)
        sc1 = _ns()
        setup_compute_s = _s(sc1 - sc0)
        if not no_setup_cache:
            sv0 = _ns()
            _save_pickle(setup_cache_path, pk)
            sv1 = _ns()
            setup_save_s = _s(sv1 - sv0)
    s1 = _ns()
    public_values = [1, y]

    pool = CPUMemoryPool()
    trace = Trace()
    runtime_attrs = {
        "fixed_base_policy": str(fixed_base_policy),
        "fixed_base_window_bits": int(fixed_base_window_bits),
        "fixed_base_auto_min_points": int(fixed_base_auto_min_points),
        "fixed_base_auto_groth16_min_calls": int(fixed_base_auto_groth16_min_calls),
    }

    t0 = _ns()
    proofs = []
    for _ in range(batch):
        proofs.append(groth16_prove(ir, pk, wit, runtime_trace=trace, runtime_pool=pool, runtime_attrs=runtime_attrs))
    t1 = _ns()

    v0 = _ns()
    oks = [groth16_verify(pk.vk, public_values, p) for p in proofs]
    v1 = _ns()

    if not all(oks):
        raise RuntimeError("groth16 verify failed")
    total1 = _ns()

    return {
        "scheme": "groth16",
        "repeat": repeat,
        "batch": batch,
        "total_s": _s(total1 - total0),
        "compile_s": _s(c1 - c0),
        "witness_s": _s(w1 - w0),
        "check_r1cs_s": _s(r1 - r0),
        "setup_s": _s(s1 - s0),
        "setup_cache_path": setup_cache_path,
        "setup_cache_hit": setup_cache_hit,
        "setup_load_s": setup_load_s,
        "setup_compute_s": setup_compute_s,
        "setup_save_s": setup_save_s,
        "warmup_fixed_base": False,
        "warmup_fixed_base_window_bits": 0,
        "warmup_fixed_base_points_n": 0,
        "warmup_fixed_base_cached": False,
        "warmup_fixed_base_s": 0.0,
        "fixed_base_policy": str(fixed_base_policy),
        "fixed_base_window_bits": int(fixed_base_window_bits),
        "fixed_base_auto_min_points": int(fixed_base_auto_min_points),
        "fixed_base_auto_groth16_min_calls": int(fixed_base_auto_groth16_min_calls),
        "prove_s": _s(t1 - t0),
        "verify_s": _s(v1 - v0),
        "trace_total_s": _s(trace.total_ns()),
        "trace_by_op_s": {k: _s(v) for k, v in trace.summarize_ns_by_op().items()},
        "pool": {
            "alloc_calls": pool.stats.alloc_calls,
            "reuse_calls": pool.stats.reuse_calls,
            "peak_in_use": pool.stats.peak_in_use,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    # 选择压测的证明系统
    p.add_argument("--scheme", choices=["plonk", "groth16"], required=True)
    # 线性放大电路规模
    p.add_argument("--repeat", type=int, default=0)
    # 连续生成证明的次数
    p.add_argument("--batch", type=int, default=1)
    # 手动设定 setup 缓存文件的保存路径
    p.add_argument("--setup-cache-path", type=str, default=None)
    # 是否禁用 setup 缓存
    p.add_argument("--no-setup-cache", action="store_true")
    # 是否预计算固定基点
    p.add_argument("--warmup-fixed-base", action="store_true")
    # 预计算固定基点的切片窗口大小
    p.add_argument("--warmup-fixed-base-window-bits", type=int, default=8)
    # 预计算固定基点的次数
    p.add_argument("--warmup-fixed-base-n", type=int, default=0)
    p.add_argument("--fixed-base-policy", choices=["off", "auto", "on"], default="off")
    p.add_argument("--fixed-base-window-bits", type=int, default=8)
    p.add_argument("--fixed-base-auto-min-points", type=int, default=256)
    p.add_argument("--fixed-base-auto-groth16-min-calls", type=int, default=2)
    args = p.parse_args()

    if args.scheme == "plonk":
        res = bench_plonk(
            repeat=args.repeat,
            batch=args.batch,
            setup_cache_path=args.setup_cache_path,
            no_setup_cache=args.no_setup_cache,
            warmup_fixed_base=args.warmup_fixed_base,
            warmup_fixed_base_window_bits=args.warmup_fixed_base_window_bits,
            warmup_fixed_base_n=args.warmup_fixed_base_n,
            fixed_base_policy=args.fixed_base_policy,
            fixed_base_window_bits=args.fixed_base_window_bits,
            fixed_base_auto_min_points=args.fixed_base_auto_min_points,
        )
    else:
        res = bench_groth16(
            repeat=args.repeat,
            batch=args.batch,
            setup_cache_path=args.setup_cache_path,
            no_setup_cache=args.no_setup_cache,
            fixed_base_policy=args.fixed_base_policy,
            fixed_base_window_bits=args.fixed_base_window_bits,
            fixed_base_auto_min_points=args.fixed_base_auto_min_points,
            fixed_base_auto_groth16_min_calls=args.fixed_base_auto_groth16_min_calls,
        )

    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
