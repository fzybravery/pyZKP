from __future__ import annotations

"""
Kernel 注册表。

核心职责：将 (OpType, Device) 映射到可执行函数 KernelFn。
上层只需要构图（op + inputs/outputs + attrs），执行时由 Executor 根据设备选择对应 kernel。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from runtime.ir.ops import OpType
from runtime.ir.types import Backend, Device

# 定义所有底层执行函数的标准签名
KernelFn = Callable[[Dict[str, Any]], Dict[str, Any]]

# 算子注册表
# 默认 backend 是 CPU，其他 backend 可以通过 register 注册。
@dataclass
class KernelRegistry:
    _kernels: Dict[Tuple[OpType, Device, Backend], KernelFn]

    def __init__(self) -> None:
        self._kernels = {}

    # 注册算子
    def register(self, op: OpType, device: Device, fn: KernelFn, *, backend: Backend = Backend.CPU) -> None:
        """
        注册某个设备上的算子实现。
        """
        key = (op, device, backend)
        if key in self._kernels:
            raise ValueError(f"kernel already registered: {op} on {device} ({backend})")
        self._kernels[key] = fn

    # 获取算子
    def get(self, op: OpType, device: Device, *, backend: Backend = Backend.CPU) -> KernelFn:
        """
        获取某个设备上的算子实现；不存在则抛错。
        """
        key = (op, device, backend)
        if key not in self._kernels:
            raise KeyError(f"kernel not found: {op} on {device} ({backend})")
        return self._kernels[key]

    def has(self, op: OpType, device: Device, *, backend: Backend = Backend.CPU) -> bool:
        """
        检查是否注册了某个算子实现。
        """
        key = (op, device, backend)
        return key in self._kernels
