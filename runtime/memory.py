from __future__ import annotations

"""
CPU 侧内存池（最小实现）。

当前仅对 FR 的 list[int] 做复用：
- 这类数组在 NTT/coset/逐点运算中最频繁、最容易造成分配压力
- 先把接口打通，为后续 GPU/跨设备内存池抽象铺路
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from pyZKP.runtime.ir.types import DType, Device


@dataclass
class PoolStats:
    """
    内存池统计信息（用于 bench/回归）。
    """
    alloc_calls: int = 0 # 外界向内存池“要”了多少次内存
    reuse_calls: int = 0 # 内存池复用了了多少次内存数组
    in_use: int = 0 # 当前内存池中正在使用的数组数量
    peak_in_use: int = 0 # 内存池中曾经使用的最大数组数量（peak）


class CPUMemoryPool:
    """
    简单的 CPU 内存池：按 (dtype, length) 复用 Python list。
    """
    def __init__(self) -> None:
        self._free: Dict[Tuple[DType, int], List[List[int]]] = {}
        self._owned_ids: set[int] = set()
        self.stats = PoolStats()

    def alloc_fr(self, n: int) -> List[int]:
        """
        分配（或复用）一个长度为 n 的 FR 数组。
        """
        return self.alloc(DType.FR, n)

    def alloc(self, dtype: DType, n: int) -> List[int]:
        """
        分配（或复用）一个长度为 n 的数组。

        返回值会被清零，保证上层不需要手动初始化。
        """
        self.stats.alloc_calls += 1
        key = (dtype, int(n))
        lst = self._free.get(key)
        if lst is not None and len(lst) > 0:
            self.stats.reuse_calls += 1
            self.stats.in_use += 1
            if self.stats.in_use > self.stats.peak_in_use:
                self.stats.peak_in_use = self.stats.in_use
            out = lst.pop()
            self._owned_ids.add(id(out))
            for i in range(len(out)):
                out[i] = 0
            return out
        self.stats.in_use += 1
        if self.stats.in_use > self.stats.peak_in_use:
            self.stats.peak_in_use = self.stats.in_use
        out = [0] * int(n)
        self._owned_ids.add(id(out))
        return out

    def release(self, dtype: DType, arr: List[int]) -> None:
        """
        归还数组到池中，供后续复用。
        """
        if id(arr) not in self._owned_ids:
            return
        self.stats.in_use -= 1
        key = (dtype, len(arr))
        self._free.setdefault(key, []).append(arr)
