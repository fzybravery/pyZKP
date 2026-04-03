from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.ir.types import Backend, Device
from runtime.metal.runtime import MetalRuntime, metal_available

# 将“执行后端信息 + 资源管理对象”等打包为一个上下文对象
# 运行时“执行环境/资源句柄”，决定最终执行环境与资源
@dataclass
class DeviceContext:
    backend: Backend = Backend.CPU
    device: Device = Device.CPU
    pool: Any | None = None
    config: Any | None = None  # Reference to RuntimeConfig

# CPU 上下文
@dataclass
class CPUContext(DeviceContext):
    backend: Backend = Backend.CPU
    device: Device = Device.CPU

# Metal 运行时资源
@dataclass
class MetalContext(DeviceContext):
    backend: Backend = Backend.METAL
    device: Device = Device.METAL
    metal: Any | None = None

    @staticmethod
    def create_default(*, pool=None, config=None) -> "MetalContext":
        if not metal_available():
            raise RuntimeError("Metal runtime not available (missing PyObjC Metal/MetalKit)")
        rt = MetalRuntime.create_default()
        return MetalContext(pool=pool, metal=rt, config=config)
