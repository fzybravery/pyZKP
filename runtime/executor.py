from __future__ import annotations

"""
运行时执行器（最小实现）。

功能：
- 对 Graph 做 analyze()，得到拓扑序并顺序执行
- 通过 KernelRegistry 根据 (op, device) 选择 kernel
- 可选 trace：记录每个算子的耗时与规模
- 可选 pool：复用 FR 数组，并在 keep 策略下回收中间 buffer
"""

from dataclasses import dataclass
import time
from typing import Any, Dict

from pyZKP.runtime.context import CPUContext, DeviceContext
from pyZKP.runtime.config import RuntimeConfig
from pyZKP.runtime.ir.graph import Graph, GraphAnalysis, Node
from pyZKP.runtime.ir.types import Backend, Buffer, Device, DType
from pyZKP.runtime.kernels.registry import KernelRegistry
from pyZKP.runtime.memory import CPUMemoryPool
from pyZKP.runtime.trace import Trace, TraceEvent

# 执行器，依赖算子注册表来获取算子具体的执行函数
@dataclass
class Executor:
    registry: KernelRegistry

    def run(
        self,
        graph: Graph,
        *,
        trace: Trace | None = None,
        pool: CPUMemoryPool | None = None,
        keep: list[str] | None = None,
        backend: Backend = Backend.CPU,
        context: DeviceContext | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        """
        执行一张算子图。

        keep：若提供，则运行过程中会回收（删除）不再被使用且不在 keep 中的中间 buffer。
        """
        analysis = graph.analyze_cached()
        if runtime_config is not None:
            backend = runtime_config.backend
        ctx = context or CPUContext(backend=backend, pool=pool)
        if ctx.pool is None:
            ctx.pool = pool
        backend = ctx.backend
        self._run_with_analysis(graph, analysis, trace=trace, pool=ctx.pool, keep=keep, backend=backend, context=ctx)

    def run_repeated(
        self,
        graph: Graph,
        repeat: int,
        *,
        trace: Trace | None = None,
        pool: CPUMemoryPool | None = None,
        keep: list[str] | None = None,
        backend: Backend = Backend.CPU,
        context: DeviceContext | None = None,
        before_each=None,
    ) -> None:
        analysis = graph.analyze_cached()
        ctx = context or CPUContext(backend=backend, pool=pool)
        if ctx.pool is None:
            ctx.pool = pool
        backend = ctx.backend
        keep0 = set(keep) if keep is not None else set()
        keep0 |= set(analysis.initial)
        for i in range(int(repeat)):
            if before_each is not None:
                before_each(i, graph)
            self._run_with_analysis(graph, analysis, trace=trace, pool=ctx.pool, keep=list(keep0), backend=backend, context=ctx)

    def _run_with_analysis(
        self,
        graph: Graph,
        analysis: GraphAnalysis,
        *,
        trace: Trace | None,
        pool: CPUMemoryPool | None,
        keep: list[str] | None,
        backend: Backend,
        context: DeviceContext,
    ) -> None:
        keep_set = set(keep) if keep is not None else None
        use_count: Dict[str, int] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                use_count[inp] = use_count.get(inp, 0) + 1

        for idx in analysis.topo_order:
            node = graph.nodes[idx]
            self._run_node(graph, node, trace=trace, pool=pool, backend=backend, context=context)
            if keep_set is not None:
                for inp in node.inputs:
                    use_count[inp] = use_count.get(inp, 0) - 1
                    if use_count[inp] == 0 and inp in graph.buffers and inp not in keep_set:
                        buf = graph.buffers[inp]
                        # 目前仅对 CPU/FR/list[int] 做内存池复用；其他类型先直接丢弃引用。
                        if pool is not None and buf.device == Device.CPU and buf.dtype == DType.FR and isinstance(buf.data, list) and hasattr(pool, "release"):
                            pool.release(DType.FR, buf.data)
                        del graph.buffers[inp]

    def _run_node(
        self,
        graph: Graph,
        node: Node,
        *,
        trace: Trace | None,
        pool: CPUMemoryPool | None,
        backend: Backend,
        context: DeviceContext,
    ) -> None:
        if len(node.inputs) == 0:
            raise ValueError("node must have inputs")
        in_buf0 = graph.buffers[node.inputs[0]]
        device: Device = in_buf0.device

        fn = self.registry.get(node.op, device, backend=backend)
        inputs = [graph.buffers[i] for i in node.inputs]
        input_sizes = [_buffer_size(b) for b in inputs]
        ctx: Dict[str, Any] = {
            "graph": graph,
            "node": node,
            "inputs": inputs,
            "attrs": dict(node.attrs),
            "pool": pool,
            "context": context,
        }
        t0 = time.perf_counter_ns()
        out = fn(ctx)
        t1 = time.perf_counter_ns()
        if "outputs" not in out:
            raise ValueError("kernel must return outputs")
        outputs: Dict[str, Buffer] = out["outputs"]
        output_sizes = [_buffer_size(outputs[k]) for k in node.outputs if k in outputs]
        for bid, buf in outputs.items():
            graph.buffers[bid] = buf

        if trace is not None:
            trace.add(
                TraceEvent(
                    op=node.op,
                    device=device,
                    backend=backend,
                    start_ns=t0,
                    end_ns=t1,
                    attrs=dict(node.attrs),
                    input_sizes=input_sizes,
                    output_sizes=output_sizes,
                )
            )


def _buffer_size(buf: Buffer) -> int:
    """
    粗略估计 buffer 的“规模”，用于 trace。
    """
    d = buf.data
    if isinstance(d, (list, tuple, bytes, bytearray)):
        return len(d)
    return 1
