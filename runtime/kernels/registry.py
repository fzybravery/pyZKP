from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from pyZKP.runtime.ir.ops import OpType
from pyZKP.runtime.ir.types import Device

# 定义所有底层执行函数的标准签名
KernelFn = Callable[[Dict[str, Any]], Dict[str, Any]]

# 算子注册表
@dataclass
class KernelRegistry:
    _kernels: Dict[Tuple[OpType, Device], KernelFn]

    def __init__(self) -> None:
        self._kernels = {}

    # 注册算子
    def register(self, op: OpType, device: Device, fn: KernelFn) -> None:
        key = (op, device)
        if key in self._kernels:
            raise ValueError(f"kernel already registered: {op} on {device}")
        self._kernels[key] = fn

    # 获取算子
    def get(self, op: OpType, device: Device) -> KernelFn:
        key = (op, device)
        if key not in self._kernels:
            raise KeyError(f"kernel not found: {op} on {device}")
        return self._kernels[key]

