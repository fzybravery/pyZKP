from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .ops import OpType
from .types import Buffer, Device, DType


# 图节点，节点只记录算子类型、输入输出、属性
@dataclass(frozen=True)
class Node:
    op: OpType
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]

# 图：承载所有 buffer 和 node
class Graph:
    def __init__(self) -> None:
        self.buffers: Dict[str, Buffer] = {}
        self.nodes: List[Node] = []

    def add_buffer(self, *, id: str, device: Device, dtype: DType, data: Any, meta: Dict[str, Any] | None = None) -> Buffer:
        if id in self.buffers:
            raise ValueError(f"buffer already exists: {id}")
        buf = Buffer(id=id, device=device, dtype=dtype, data=data, meta=meta)
        self.buffers[id] = buf
        return buf

    # 添加节点，后续执行时根据节点顺序执行
    def add_node(self, *, op: OpType, inputs: Sequence[str], outputs: Sequence[str], attrs: Dict[str, Any] | None = None) -> Node:
        n = Node(op=op, inputs=list(inputs), outputs=list(outputs), attrs=attrs or {})
        self.nodes.append(n)
        return n

