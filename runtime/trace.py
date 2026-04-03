from __future__ import annotations

"""
运行时 Trace/Profiler（最小实现）。

目标：
- 记录每个算子（op）的执行耗时与输入输出规模
- 支持按 op 聚合，为 benchmark/回归提供可观测性
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from runtime.ir.ops import OpType
from runtime.ir.types import Backend, Device


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

    def clear(self) -> None:
        self.events.clear()

    def export_chrome_tracing(self, filepath: str) -> None:
        """
        导出为 Chrome Tracing (JSON) 格式，可以在 chrome://tracing 或 edge://tracing 中查看
        """
        import json
        import os
        events = []
        
        # 将不同后端映射到不同的 PID (Process ID) 以产生泳道效果
        pid_map = {
            "CPU": 1,
            "METAL": 2
        }
        
        # 添加进程元数据 (进程名称)
        events.append({
            "name": "process_name", "ph": "M", "pid": pid_map["CPU"],
            "args": {"name": "CPU Backend"}
        })
        events.append({
            "name": "process_name", "ph": "M", "pid": pid_map["METAL"],
            "args": {"name": "Metal Backend"}
        })

        for trace in self.events:
            # Chrome Tracing 时间单位是微秒 (microseconds)
            start_us = trace.start_ns / 1000.0
            dur_us = trace.duration_ns / 1000.0
            
            # 使用大写后端名称获取对应的 PID，如果没有找到默认用 CPU
            backend_str = str(trace.backend).split('.')[-1].upper()
            pid = pid_map.get(backend_str, 1)
            
            event = {
                "name": str(trace.op.value),
                "cat": "Op",
                "ph": "X",  # Complete event (has duration)
                "ts": start_us,
                "dur": dur_us,
                "pid": pid,
                "tid": 1,   # 统一在主线程显示
                "args": {
                    "backend": str(trace.backend),
                    "device": str(trace.device),
                    "input_sizes": trace.input_sizes,
                    "output_sizes": trace.output_sizes,
                    **trace.attrs
                }
            }
            events.append(event)
            
        # 确保目录存在
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f, indent=2)


# Global tracer instance for use in benchmarks
tracer = Trace()
