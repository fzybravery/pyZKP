from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from pyZKP.runtime.ir.graph import Graph, Node
from pyZKP.runtime.ir.types import Buffer, Device
from pyZKP.runtime.kernels.registry import KernelRegistry

# 执行器，依赖算子注册表来获取算子具体的执行函数
@dataclass
class Executor:
    registry: KernelRegistry

    def run(self, graph: Graph) -> None:
        for node in graph.nodes:
            self._run_node(graph, node)

    def _run_node(self, graph: Graph, node: Node) -> None:
        if len(node.inputs) == 0:
            raise ValueError("node must have inputs")
        in_buf0 = graph.buffers[node.inputs[0]]
        device: Device = in_buf0.device

        fn = self.registry.get(node.op, device)
        ctx: Dict[str, Any] = {
            "graph": graph,
            "node": node,
            "inputs": [graph.buffers[i] for i in node.inputs],
            "attrs": dict(node.attrs),
        }
        out = fn(ctx)
        if "outputs" not in out:
            raise ValueError("kernel must return outputs")
        outputs: Dict[str, Buffer] = out["outputs"]
        for bid, buf in outputs.items():
            graph.buffers[bid] = buf

