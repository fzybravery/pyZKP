from __future__ import annotations

"""
Kernel 注册表。

核心职责：将 (OpType, Device) 映射到可执行函数 KernelFn。
上层只需要构图（op + inputs/outputs + attrs），执行时由 Executor 根据设备选择对应 kernel。
"""

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
        """
        注册某个设备上的算子实现。
        """
        key = (op, device)
        if key in self._kernels:
            raise ValueError(f"kernel already registered: {op} on {device}")
        self._kernels[key] = fn

    # 获取算子
    def get(self, op: OpType, device: Device) -> KernelFn:
        """
        获取某个设备上的算子实现；不存在则抛错。
        """
        key = (op, device)
        if key not in self._kernels:
            raise KeyError(f"kernel not found: {op} on {device}")
        return self._kernels[key]
