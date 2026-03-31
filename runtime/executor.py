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
from pyZKP.runtime.ir.ops import OpType
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
        if context is not None:
            ctx = context
        elif runtime_config is not None:
            ctx = runtime_config.make_context(pool=pool)
        else:
            ctx = CPUContext(backend=backend, pool=pool)
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

        # 在执行前，进行自动异构图重写 (Auto Graph Rewrite Pass)
        # 这一步负责根据 backend 动态插入 TO_DEVICE 和 FROM_DEVICE 算子
        if backend != Backend.CPU:
            # 由于重写图会修改 graph.nodes 和 graph.buffers，我们需要一个动态的顺序
            # 简化起见，这里我们在原本的 topo_order 执行过程中，遇到跨设备边界时，直接临时插入转换节点并立即执行它。
            pass

        for idx in analysis.topo_order:
            node = graph.nodes[idx]
            
            # --- 自动异构设备切分与图重写 ---
            # 决定当前节点要在哪个设备上执行
            target_device = Device.CPU
            if backend != Backend.CPU and self.registry.has(node.op, context.device, backend=backend):
                target_device = context.device
            
            # 检查所有的输入 Buffer 是否都在 target_device 上
            # 如果不在，我们需要自动插入 TO_DEVICE 或 FROM_DEVICE
            new_inputs = []
            for inp_id in node.inputs:
                inp_buf = graph.buffers[inp_id]
                if inp_buf.device != target_device:
                    # 需要进行设备间搬运
                    transfer_op = OpType.TO_DEVICE if target_device != Device.CPU else OpType.FROM_DEVICE
                    transfer_out_id = f"{inp_id}_{target_device.value}"
                    
                    # 如果之前还没搬运过这个 buffer，执行搬运
                    if transfer_out_id not in graph.buffers:
                        transfer_node = Node(op=transfer_op, inputs=[inp_id], outputs=[transfer_out_id], attrs={})
                        self._run_node(graph, transfer_node, trace=trace, pool=pool, backend=backend, context=context, force_device=Device.CPU) # 搬运算子(TO_DEVICE/FROM_DEVICE)本身通常注册在 CPU device 下
                    
                    new_inputs.append(transfer_out_id)
                else:
                    new_inputs.append(inp_id)
            
            # 更新当前节点的输入
            # Node 是 dataclass(frozen=True)，我们需要创建一个新对象或在 _run_node 中传递覆盖
            if new_inputs != node.inputs:
                node = Node(op=node.op, inputs=new_inputs, outputs=node.outputs, attrs=node.attrs)
                graph.nodes[idx] = node
            
            # 运行当前节点
            self._run_node(graph, node, trace=trace, pool=pool, backend=backend, context=context, force_device=target_device)
            # --- 自动重写结束 ---
            
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
        force_device: Device | None = None,
    ) -> None:
        if len(node.inputs) == 0:
            raise ValueError("node must have inputs")
        in_buf0 = graph.buffers[node.inputs[0]]
        
        # 允许通过 force_device 覆盖基于输入的默认设备推断
        device: Device = force_device if force_device is not None else in_buf0.device

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
