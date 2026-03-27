from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# 设备类型
class Device(str, Enum):
    CPU = "cpu"

# 后端类型
class Backend(str, Enum):
    CPU = "cpu"
    METAL = "metal"

# 缓冲区中存储的数据类型
class DType(str, Enum):
    FR = "fr"
    G1 = "g1"
    G2 = "g2"
    BYTES = "bytes"
    OBJ = "obj"

# 定义“运行时”的最小抽象
@dataclass
class Buffer:
    id: str
    device: Device
    dtype: DType
    data: Any
    meta: Optional[Dict[str, Any]] = None
