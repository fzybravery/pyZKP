from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Tuple

from common.ir.core import Input, Var, VarRef, Visibility

# 记录标签的可见性
@dataclass(frozen=True)
class InputSpec:
    visibility: Visibility
    name: str | None = None


def public(name: str | None = None) -> InputSpec:
    return InputSpec(visibility=Visibility.PUBLIC, name=name)


def secret(name: str | None = None) -> InputSpec:
    return InputSpec(visibility=Visibility.SECRET, name=name)

# 提取属性及其对应值
def _iter_fields(obj: Any) -> List[Tuple[str, Any]]:
    if is_dataclass(obj):
        return [(f.name, getattr(obj, f.name)) for f in fields(obj)]
    if hasattr(obj, "__dict__"):
        return list(obj.__dict__.items())
    return []

# 遍历电路，收集所有的变量标记，并且分配连续的变量ID
def walk_and_allocate_inputs(circuit: Any, start_var_id: int = 0) -> Tuple[List[Input], List[Var], Dict[str, VarRef], int]:
    inputs: List[Input] = []
    vars_: List[Var] = []
    env: Dict[str, VarRef] = {}
    next_id = start_var_id

    seen_names: Dict[str, int] = {}

    def alloc_one(field_path: str, spec: InputSpec) -> VarRef:
        nonlocal next_id
        name = spec.name or field_path
        if name in seen_names:
            raise ValueError(f"duplicate input name: {name}")
        seen_names[name] = 1
        inputs.append(Input(name=name, visibility=spec.visibility))
        vars_.append(Var(id=next_id, name=name, visibility=spec.visibility))
        ref = VarRef(next_id)
        env[name] = ref
        next_id += 1
        return ref

    to_alloc: List[Tuple[str, Any, str, InputSpec]] = []

    def collect(obj: Any, prefix: str) -> None:
        for k, v in _iter_fields(obj):
            field_path = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
            if isinstance(v, InputSpec):
                to_alloc.append((field_path, obj, k, v))
                continue
            if is_dataclass(v) or hasattr(v, "__dict__"):
                collect(v, field_path)

    collect(circuit, "")

    for field_path, obj, k, spec in [t for t in to_alloc if t[3].visibility == Visibility.PUBLIC]:
        ref = alloc_one(field_path, spec)
        try:
            setattr(obj, k, ref)
        except Exception:
            pass

    for field_path, obj, k, spec in [t for t in to_alloc if t[3].visibility == Visibility.SECRET]:
        ref = alloc_one(field_path, spec)
        try:
            setattr(obj, k, ref)
        except Exception:
            pass

    return inputs, vars_, env, next_id
