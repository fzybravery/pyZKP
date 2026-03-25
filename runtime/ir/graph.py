from __future__ import annotations

"""
运行时算子图（IR）。

Graph 保存两类对象：
- Buffer：数据节点（由外部注入或由算子节点产出）
- Node：算子节点（op + inputs/outputs + attrs）

Graph.analyze() 负责：
- 静态合法性校验（输入是否存在、输出是否重复定义、是否覆盖初始 buffer）
- 构建依赖并给出拓扑序（Executor 可据此按 DAG 顺序执行）
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .ops import OpType
from .types import Buffer, Device, DType


@dataclass(frozen=True)
class Node:
    """
    算子节点（纯描述，不包含实际数据）。

    - inputs/outputs 存的是 buffer id
    - attrs 存的是算子参数（如 n/omega/shift、挑战值等）
    """
    op: OpType
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class GraphAnalysis:
    """
    图分析结果。

    - producers: 每一个buffer id是由哪个节点产出的
    - topo_order: 拓扑排序后的node执行顺序（这里的标号指的是node在Graph的nodes中的位置索引）
    """
    producers: Dict[str, int]
    topo_order: Tuple[int, ...]


class Graph:
    """
    最小可用的算子图容器。

    当前 Graph 允许节点以任意顺序插入，但 Executor 会基于 analyze() 的拓扑序执行，
    因此构图端不需要手工维护“先后顺序”。
    """
    def __init__(self) -> None:
        self.buffers: Dict[str, Buffer] = {}
        self.nodes: List[Node] = []

    def add_buffer(self, *, id: str, device: Device, dtype: DType, data: Any, meta: Dict[str, Any] | None = None) -> Buffer:
        """
        向图中注入一个初始 buffer（通常是常量、输入或外部对象）。
        """
        if id in self.buffers:
            raise ValueError(f"buffer already exists: {id}")
        buf = Buffer(id=id, device=device, dtype=dtype, data=data, meta=meta)
        self.buffers[id] = buf
        return buf

    def add_node(self, *, op: OpType, inputs: Sequence[str], outputs: Sequence[str], attrs: Dict[str, Any] | None = None) -> Node:
        """
        添加一个算子节点。

        注意：节点插入顺序不必严格符合依赖顺序；Executor 会按拓扑序执行。
        """
        n = Node(op=op, inputs=list(inputs), outputs=list(outputs), attrs=attrs or {})
        self.nodes.append(n)
        return n

    def analyze(self) -> GraphAnalysis:
        """
        对当前图做静态检查与依赖分析，并返回拓扑执行顺序。
        """
        producers: Dict[str, int] = {}
        initial = set(self.buffers.keys())

        for idx, node in enumerate(self.nodes):
            if len(node.outputs) == 0:
                raise ValueError(f"node has no outputs: {idx}")
            for out in node.outputs:
                if out in initial:
                    raise ValueError(f"node output overwrites existing buffer: {out}")
                if out in producers:
                    raise ValueError(f"buffer produced by multiple nodes: {out}")
                producers[out] = idx

        indeg = [0] * len(self.nodes)
        succ: List[List[int]] = [[] for _ in range(len(self.nodes))]

        for j, node in enumerate(self.nodes):
            if len(node.inputs) == 0:
                raise ValueError(f"node has no inputs: {j}")
            for inp in node.inputs:
                if inp in producers:
                    i = producers[inp]
                    succ[i].append(j)
                    indeg[j] += 1
                elif inp in initial:
                    continue
                else:
                    raise ValueError(f"missing input buffer: {inp}")

        # Kahn 算法：对 DAG 做拓扑排序
        q = [i for i, d in enumerate(indeg) if d == 0]
        topo: List[int] = []
        while len(q) > 0:
            i = q.pop()
            topo.append(i)
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)

        if len(topo) != len(self.nodes):
            raise ValueError("graph has cycle or unresolved dependencies")

        return GraphAnalysis(producers=producers, topo_order=tuple(topo))
