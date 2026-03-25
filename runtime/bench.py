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
import time

from pyZKP import build_witness, check_r1cs, compile_circuit
from pyZKP.backend.schemes.groth16.prove import prove as groth16_prove
from pyZKP.backend.schemes.groth16.setup import setup as groth16_setup
from pyZKP.backend.schemes.groth16.verify import verify as groth16_verify
from pyZKP.backend.schemes.plonk import setup as plonk_setup
from pyZKP.backend.schemes.plonk.prove import prove as plonk_prove
from pyZKP.backend.schemes.plonk.verify import verify as plonk_verify
from pyZKP.common.crypto.field.fr import FR_MODULUS
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


def bench_plonk(*, repeat: int, batch: int) -> dict:
    """
    运行 PLONK 的 setup/prove/verify，并返回性能统计。
    """
    ir = compile_circuit(RepeatMulCircuit(repeat), FR_MODULUS)
    x = 9
    y = (x * x * x + x + 5) % FR_MODULUS
    wit = build_witness(ir, {"x": x, "y": y})
    check_r1cs(ir, wit)

    pk = plonk_setup(ir)
    public_values = [1, y]

    pool = CPUMemoryPool()
    trace = Trace()

    t0 = _ns()
    proofs = []
    for _ in range(batch):
        proofs.append(plonk_prove(pk, wit, public_values=public_values, runtime_trace=trace, runtime_pool=pool))
    t1 = _ns()

    v0 = _ns()
    oks = [plonk_verify(pk.vk, p, public_values=public_values) for p in proofs]
    v1 = _ns()

    if not all(oks):
        raise RuntimeError("plonk verify failed")

    return {
        "scheme": "plonk",
        "repeat": repeat,
        "batch": batch,
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


def bench_groth16(*, repeat: int, batch: int) -> dict:
    """
    运行 Groth16 的 setup/prove/verify，并返回性能统计。
    """
    ir = compile_circuit(RepeatMulCircuit(repeat), FR_MODULUS)
    x = 9
    y = (x * x * x + x + 5) % FR_MODULUS
    wit = build_witness(ir, {"x": x, "y": y})
    check_r1cs(ir, wit)

    pk = groth16_setup(ir)
    public_values = [1, y]

    pool = CPUMemoryPool()
    trace = Trace()

    t0 = _ns()
    proofs = []
    for _ in range(batch):
        proofs.append(groth16_prove(ir, pk, wit, runtime_trace=trace, runtime_pool=pool))
    t1 = _ns()

    v0 = _ns()
    oks = [groth16_verify(pk.vk, public_values, p) for p in proofs]
    v1 = _ns()

    if not all(oks):
        raise RuntimeError("groth16 verify failed")

    return {
        "scheme": "groth16",
        "repeat": repeat,
        "batch": batch,
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
    p.add_argument("--scheme", choices=["plonk", "groth16"], required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    if args.scheme == "plonk":
        res = bench_plonk(repeat=args.repeat, batch=args.batch)
    else:
        res = bench_groth16(repeat=args.repeat, batch=args.batch)

    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
