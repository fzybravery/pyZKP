from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from runtime.ir.types import DType, Device


@dataclass
class PoolStats:
    """
    内存池统计信息（用于 bench/回归）。
    """
    alloc_calls: int = 0 # 外界向内存池“要”了多少次内存
    reuse_calls: int = 0 # 内存池复用了了多少次内存数组
    in_use: int = 0 # 当前内存池中正在使用的数组数量
    peak_in_use: int = 0 # 内存池中曾经使用的最大数组数量（peak）


class MemoryPool:
    """
    统一内存池：复用 CPU 的 list[int] 以及 UMA 下 Metal 的 MTLBuffer。
    """
    def __init__(self) -> None:
        # (数据类型， 数组长度) : 可用id列表
        self._free_cpu: Dict[Tuple[DType, int], List[List[int]]] = {}
        self._owned_cpu_ids: set[int] = set() # 已被分配的 CPU 数组id
        self.cpu_stats = PoolStats() # CPU 内存池统计信息

        # (字节数) : 可用MTLBuffer列表
        self._free_metal: Dict[int, List[Any]] = {} # byte_size -> List[MTLBuffer]
        self._owned_metal_ids: set[int] = set()
        self.metal_stats = PoolStats()

    def alloc_fr(self, n: int) -> List[int]:
        """
        分配（或复用）一个长度为 n 的 CPU FR 数组。
        """
        return self.alloc_cpu(DType.FR, n)

    def alloc_cpu(self, dtype: DType, n: int) -> List[int]:
        """
        分配（或复用）一个长度为 n 的 CPU 数组。
        """
        self.cpu_stats.alloc_calls += 1
        key = (dtype, int(n))
        lst = self._free_cpu.get(key)
        if lst is not None and len(lst) > 0:
            self.cpu_stats.reuse_calls += 1
            self.cpu_stats.in_use += 1
            if self.cpu_stats.in_use > self.cpu_stats.peak_in_use:
                self.cpu_stats.peak_in_use = self.cpu_stats.in_use
            out = lst.pop()
            self._owned_cpu_ids.add(id(out))
            for i in range(len(out)):
                out[i] = 0
            return out
        self.cpu_stats.in_use += 1
        if self.cpu_stats.in_use > self.cpu_stats.peak_in_use:
            self.cpu_stats.peak_in_use = self.cpu_stats.in_use
        out = [0] * int(n)
        self._owned_cpu_ids.add(id(out))
        return out

    def release_cpu(self, dtype: DType, arr: List[int]) -> None:
        """
        归还 CPU 数组到池中，供后续复用。
        """
        if id(arr) not in self._owned_cpu_ids:
            return
        self.cpu_stats.in_use -= 1
        key = (dtype, len(arr))
        self._free_cpu.setdefault(key, []).append(arr)

    def alloc_metal(self, rt: Any, size_bytes: int) -> Any:
        """
        分配（或复用）一个大小为 size_bytes 的共享 Metal Buffer。
        零拷贝设计的核心：直接在 UMA 上分配，并保留引用以便复用。
        """
        self.metal_stats.alloc_calls += 1
        lst = self._free_metal.get(size_bytes)
        if lst is not None and len(lst) > 0:
            self.metal_stats.reuse_calls += 1
            self.metal_stats.in_use += 1
            if self.metal_stats.in_use > self.metal_stats.peak_in_use:
                self.metal_stats.peak_in_use = self.metal_stats.in_use
            out = lst.pop()
            self._owned_metal_ids.add(id(out))
            return out
            
        self.metal_stats.in_use += 1
        if self.metal_stats.in_use > self.metal_stats.peak_in_use:
            self.metal_stats.peak_in_use = self.metal_stats.in_use
        
        # 0 = MTLResourceStorageModeShared，真正实现 Apple Silicon 零拷贝
        out = rt.device.newBufferWithLength_options_(size_bytes, 0)
        if out is None:
            raise RuntimeError(f"failed to allocate MTLBuffer of size {size_bytes}")
        self._owned_metal_ids.add(id(out))
        return out

    def release_metal(self, mtl_buffer: Any) -> None:
        """
        归还 Metal Buffer 到池中，供后续复用。
        """
        if id(mtl_buffer) not in self._owned_metal_ids:
            return
        self.metal_stats.in_use -= 1
        size = mtl_buffer.length()
        self._free_metal.setdefault(size, []).append(mtl_buffer)

# 为了向后兼容，暂时别名 CPUMemoryPool
CPUMemoryPool = MemoryPool
