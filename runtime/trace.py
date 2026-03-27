from __future__ import annotations

"""
运行时 Trace/Profiler（最小实现）。

目标：
- 记录每个算子（op）的执行耗时与输入输出规模
- 支持按 op 聚合，为 benchmark/回归提供可观测性
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pyZKP.runtime.ir.ops import OpType
from pyZKP.runtime.ir.types import Backend, Device


@dataclass(frozen=True)
class TraceEvent:
    """
    一条算子执行事件。

    input_sizes/output_sizes 是“规模指标”，当前用 len() 粗略估计，便于后续扩展到 bytes/显存占用等。
    """
    op: OpType
    device: Device
    backend: Backend
    start_ns: int
    end_ns: int
    attrs: Dict[str, Any]
    input_sizes: List[int]
    output_sizes: List[int]

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class Trace:
    """
    Trace 容器：保存所有事件，并提供聚合统计接口。
    """
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []

    def add(self, ev: TraceEvent) -> None:
        self.events.append(ev)

    def summarize_ns_by_op(self) -> Dict[str, int]:
        """
        按 op 聚合耗时（纳秒）。
        """
        acc: Dict[str, int] = {}
        for e in self.events:
            k = str(e.op.value)
            acc[k] = acc.get(k, 0) + e.duration_ns
        return acc


    # 对每个op进行统计
    def summarize_stats_by_op(self) -> Dict[str, Dict[str, Any]]:
        acc: Dict[str, Dict[str, Any]] = {}
        for e in self.events:
            k = str(e.op.value)
            d = e.duration_ns
            inp = sum(e.input_sizes)
            out = sum(e.output_sizes)
            s = acc.get(k)
            if s is None:
                s = {
                    "count": 0,
                    "total_ns": 0,
                    "max_ns": 0,
                    "total_input_size": 0,
                    "max_input_size": 0,
                    "total_output_size": 0,
                    "max_output_size": 0,
                }
                acc[k] = s
            s["count"] += 1
            s["total_ns"] += d
            if d > s["max_ns"]:
                s["max_ns"] = d
            s["total_input_size"] += inp
            if inp > s["max_input_size"]:
                s["max_input_size"] = inp
            s["total_output_size"] += out
            if out > s["max_output_size"]:
                s["max_output_size"] = out
        for k, s in acc.items():
            c = int(s["count"])
            s["avg_ns"] = int(s["total_ns"]) // c if c else 0
            s["avg_input_size"] = float(s["total_input_size"]) / c if c else 0.0
            s["avg_output_size"] = float(s["total_output_size"]) / c if c else 0.0
        return acc

    def total_ns(self) -> int:
        """
        所有事件的总耗时（纳秒）。
        """
        return sum(e.duration_ns for e in self.events)
