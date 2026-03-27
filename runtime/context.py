from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyZKP.runtime.ir.types import Backend, Device

# 将“执行后端信息 + 资源管理对象”等打包为一个上下文对象
# 运行时“执行环境/资源句柄”，决定最终执行环境与资源
@dataclass
class DeviceContext:
    backend: Backend = Backend.CPU
    device: Device = Device.CPU
    pool: Any | None = None

# CPU 上下文
@dataclass
class CPUContext(DeviceContext):
    backend: Backend = Backend.CPU
    device: Device = Device.CPU

