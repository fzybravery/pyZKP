from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


HintFn = Callable[..., Any]


@dataclass
class HintRegistry:
    _fns: Dict[str, HintFn]

    def __init__(self) -> None:
        self._fns = {}

    def register(self, name: str, fn: HintFn) -> None:
        if not name:
            raise ValueError("hint name must be non-empty")
        if name in self._fns and self._fns[name] is not fn:
            raise ValueError(f"hint already registered: {name}")
        self._fns[name] = fn

    def get(self, name: str) -> Optional[HintFn]:
        return self._fns.get(name)


GLOBAL_HINTS = HintRegistry()
