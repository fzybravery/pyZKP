from __future__ import annotations

from typing import Any

from frontend.ir.core import CircuitIR, Field, Visibility
from frontend.api.api import API, _Builder
from frontend.circuit.schema import walk_and_allocate_inputs

# 编译电路，将电路转换为 IR 表示
def compile_circuit(circuit: Any, field_modulus: int) -> CircuitIR:
    f = Field(modulus=int(field_modulus))

    inputs, vars_, _, next_id = walk_and_allocate_inputs(circuit, start_var_id=0)

    b = _Builder(field=f, next_var_id=next_id, vars=list(vars_), constraints=[], hints=[])
    api = API(b)

    if not hasattr(circuit, "define"):
        raise TypeError("circuit must define method define(self, api)")

    circuit.define(api)

    return CircuitIR(
        field=f,
        inputs=tuple(inputs),
        vars=tuple(b.vars),
        constraints=tuple(b.constraints),
        hints=tuple(b.hints),
    )
