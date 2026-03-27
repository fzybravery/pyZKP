from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, Tuple

from pyZKP.common.ir.core.model import CircuitIR, Hint, LinExpr, VarRef

SETUP_CACHE_FORMAT_VERSION = 1


class CacheMismatchError(ValueError):
    pass

# 将 LinExpr 转换为 JSON 可序列化对象
def _linexpr_obj(e: LinExpr) -> Dict[str, Any]:
    return {"const": int(e.const), "terms": [[int(vid), int(c)] for vid, c in e.terms]}

# 序列化通用表达式，根据对象类型添加标签 t
def _expr_obj(x) -> Any:
    if isinstance(x, int):
        return {"t": "int", "v": int(x)}
    if isinstance(x, VarRef):
        return {"t": "var", "id": int(x.id)}
    if isinstance(x, LinExpr):
        return {"t": "lin", "v": _linexpr_obj(x)}
    raise TypeError(f"unknown expr: {type(x).__name__}")

# 序列化 hint，根据对象类型添加标签 op
def _hint_obj(h: Hint) -> Dict[str, Any]:
    return {"op": str(h.op), "inputs": [_expr_obj(x) for x in h.inputs], "outputs": [int(x) for x in h.outputs]}

# 计算电路 IR 的指纹，用于setup cache 校验/命名
def circuit_ir_fingerprint(ir: CircuitIR) -> str:
    obj = {
        "field_modulus": int(ir.field.modulus),
        "inputs": [{"name": i.name, "visibility": str(i.visibility.value)} for i in ir.inputs],
        "vars": [{"id": int(v.id), "name": v.name, "visibility": str(v.visibility.value)} for v in ir.vars],
        "constraints": [
            {"a": _linexpr_obj(c.a), "b": _linexpr_obj(c.b), "c": _linexpr_obj(c.c)} for c in ir.constraints
        ],
        "hints": [_hint_obj(h) for h in ir.hints],
    }
    b = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(b).hexdigest()[:16]

# 原子保存 pickle 对象
def _save_pickle(path: str, obj) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

# 原子加载 pickle 对象
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# 默认 setup cache 路径
def default_setup_cache_path(*, cache_dir: str, scheme: str, ir: CircuitIR) -> str:
    fp = circuit_ir_fingerprint(ir)
    return os.path.join(str(cache_dir), "setup", str(scheme), f"ir_{fp}_mod{int(ir.field.modulus)}_v{SETUP_CACHE_FORMAT_VERSION}.pkl")

# 保存 setup cache
def save_setup_cache(path: str, *, scheme: str, ir: CircuitIR, data) -> Dict[str, Any]:
    meta = {
        "format": "setup_cache",
        "version": int(SETUP_CACHE_FORMAT_VERSION),
        "scheme": str(scheme),
        "field_modulus": int(ir.field.modulus),
        "ir_fingerprint": circuit_ir_fingerprint(ir),
    }
    _save_pickle(path, {"meta": meta, "data": data})
    return meta

# 加载 setup cache
def load_setup_cache(path: str, *, scheme: str, ir: CircuitIR):
    obj = _load_pickle(path)
    if isinstance(obj, dict) and "meta" in obj and "data" in obj:
        meta = obj["meta"]
        if meta.get("format") != "setup_cache":
            raise CacheMismatchError("setup cache format mismatch")
        if int(meta.get("version", -1)) != int(SETUP_CACHE_FORMAT_VERSION):
            raise CacheMismatchError("setup cache version mismatch")
        if str(meta.get("scheme")) != str(scheme):
            raise CacheMismatchError("setup cache scheme mismatch")
        if int(meta.get("field_modulus", -1)) != int(ir.field.modulus):
            raise CacheMismatchError("setup cache modulus mismatch")
        if str(meta.get("ir_fingerprint")) != circuit_ir_fingerprint(ir):
            raise CacheMismatchError("setup cache circuit mismatch")
        return obj["data"], meta, True
    return obj, None, False

