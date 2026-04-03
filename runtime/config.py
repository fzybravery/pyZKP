from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from runtime.context import CPUContext, DeviceContext, MetalContext
from runtime.ir.types import Backend

# 运行时“策略/开关/默认参数集合”，其决定默认策略与默认backend
@dataclass(frozen=True)
class RuntimeConfig:
    backend: Backend = Backend.CPU
    cache_dir: str = ".pyZKP_cache"

    fixed_base_policy: str = "off"
    fixed_base_window_bits: int = 8
    fixed_base_auto_min_points: int = 256
    fixed_base_auto_groth16_min_calls: int = 2

    reuse_graph: bool = False
    reuse_prove_batch: bool = False

    # 将fixed_base等策略转换为运行时参数（attrs dict）
    def runtime_attrs(self) -> Dict[str, Any]:
        return {
            "fixed_base_policy": str(self.fixed_base_policy),
            "fixed_base_window_bits": int(self.fixed_base_window_bits),
            "fixed_base_auto_min_points": int(self.fixed_base_auto_min_points),
            "fixed_base_auto_groth16_min_calls": int(self.fixed_base_auto_groth16_min_calls),
        }

    # 在默认 attrs 上叠加调用方临时覆盖的参数
    def with_overrides(self, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
        if overrides is None:
            return self.runtime_attrs()
        out = dict(self.runtime_attrs())
        out.update(dict(overrides))
        return out

    # 创建运行时上下文
    # 用于在证明/验证时传递内存池、缓存目录等参数
    # 优先使用传入的上下文，否则创建默认上下文
    def make_context(self, *, pool=None, context: DeviceContext | None = None) -> DeviceContext:
        if context is not None:
            return context
        if self.backend == Backend.METAL:
            return MetalContext.create_default(pool=pool)
        return CPUContext(pool=pool, backend=self.backend)
