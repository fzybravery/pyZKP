from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Metal 缓冲区。GPU上的一段连续内存+元信息
@dataclass
class MetalBuffer:
    dtype: str
    n: int
    mtl_buffer: Any
    host_cache: Optional[list[int]] = None

    def __len__(self) -> int:
        return int(self.n)
