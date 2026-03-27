from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.poly import (
    coeffs_from_evals_on_roots,
    poly_div_by_xn_minus_1,
    poly_divmod,
    poly_mul,
    poly_mul_ntt,
    poly_sub,
    poly_vanishing_from_roots,
    lagrange_interpolate,
)
from pyZKP.runtime import Executor, KernelRegistry
from pyZKP.runtime.config import RuntimeConfig
from pyZKP.runtime.ir import Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels


@dataclass(frozen=True)
class QAPWitnessPolys:
    a_poly: List[int] # A(x)多项式系数
    b_poly: List[int] # B(x)多项式系数
    c_poly: List[int] # C(x)多项式系数
    h_poly: List[int] # 商多项式H(x)多项式系数
    t_poly: List[int] # 消失多项式T(x)多项式系数

# 朴素版本 qap 计算
def compute_h_from_abc(xs: Sequence[int], a_eval: Sequence[int], b_eval: Sequence[int], c_eval: Sequence[int]) -> QAPWitnessPolys:
    if not (len(xs) == len(a_eval) == len(b_eval) == len(c_eval)):
        raise ValueError("length mismatch")
    a_poly = lagrange_interpolate(xs, list(a_eval))
    b_poly = lagrange_interpolate(xs, list(b_eval))
    c_poly = lagrange_interpolate(xs, list(c_eval))
    t_poly = poly_vanishing_from_roots(xs)
    p_poly = poly_sub(poly_mul(a_poly, b_poly), c_poly)
    q, r = poly_divmod(p_poly, t_poly)
    if len(r) != 0:
        raise ValueError("witness does not satisfy R1CS (non-zero remainder)")
    return QAPWitnessPolys(a_poly=a_poly, b_poly=b_poly, c_poly=c_poly, h_poly=q, t_poly=t_poly)

"""
在单位根域上高效计算 QAP (Quadratic Arithmetic Program) 多项式。
利用 Runtime 计算图引擎，执行以下优化算法：
1. iNTT (O(n log n)): 将 a, b, c 的点值表示转换为多项式系数 A(x), B(x), C(x)。
2. NTT 乘法 (O(n log n)): 计算 P(x) = A(x)*B(x) - C(x)。
3. 特殊除法 (O(n)): 计算商多项式 h(x) = P(x) / (x^n - 1)。
如果计算出的余数不为 0，则抛出异常表示 R1CS 约束未被满足。
"""
def compute_h_from_abc_on_roots(
    n: int,
    omega: int,
    a_eval: Sequence[int],
    b_eval: Sequence[int],
    c_eval: Sequence[int],
    *,
    runtime_trace=None,
    runtime_pool=None,
    runtime_context=None,
    runtime_config: RuntimeConfig | None = None,
) -> QAPWitnessPolys:
    if not (len(a_eval) == len(b_eval) == len(c_eval) == n):
        raise ValueError("length mismatch")
    reg = KernelRegistry()
    from pyZKP.runtime.ir import Backend

    backend0 = runtime_config.backend if runtime_config is not None else Backend.CPU
    if runtime_context is not None:
        backend0 = runtime_context.backend
    register_cpu_kernels(reg, backend=backend0)
    exe = Executor(registry=reg)
    g = Graph()

    g.add_buffer(id="a_eval", device=Device.CPU, dtype=DType.FR, data=list(a_eval))
    g.add_buffer(id="b_eval", device=Device.CPU, dtype=DType.FR, data=list(b_eval))
    g.add_buffer(id="c_eval", device=Device.CPU, dtype=DType.FR, data=list(c_eval))

    g.add_node(op=OpType.ROOTS_COEFFS_FROM_EVALS, inputs=["a_eval"], outputs=["a_coeff"], attrs={"omega": omega})
    g.add_node(op=OpType.ROOTS_COEFFS_FROM_EVALS, inputs=["b_eval"], outputs=["b_coeff"], attrs={"omega": omega})
    g.add_node(op=OpType.ROOTS_COEFFS_FROM_EVALS, inputs=["c_eval"], outputs=["c_coeff"], attrs={"omega": omega})
    g.add_node(op=OpType.POLY_MUL_NTT, inputs=["a_coeff", "b_coeff"], outputs=["ab_coeff"])
    g.add_node(op=OpType.POLY_SUB, inputs=["ab_coeff", "c_coeff"], outputs=["p_coeff"])
    g.add_node(op=OpType.DIV_XN_MINUS_1, inputs=["p_coeff"], outputs=["h_coeff", "rem"], attrs={"n": n})

    ctx0 = runtime_config.make_context(pool=runtime_pool, context=runtime_context) if runtime_config is not None else runtime_context
    exe.run(g, pool=runtime_pool, trace=runtime_trace, keep=["a_coeff", "b_coeff", "c_coeff", "h_coeff", "rem"], context=ctx0)
    a_poly = list(g.buffers["a_coeff"].data)
    b_poly = list(g.buffers["b_coeff"].data)
    c_poly = list(g.buffers["c_coeff"].data)
    h_poly = list(g.buffers["h_coeff"].data)
    rem = list(g.buffers["rem"].data)
    if len(rem) != 0:
        raise ValueError("witness does not satisfy R1CS (non-zero remainder)")
    t_poly = [(-1) % FR_MODULUS] + [0] * (n - 1) + [1]
    return QAPWitnessPolys(a_poly=a_poly, b_poly=b_poly, c_poly=c_poly, h_poly=h_poly, t_poly=t_poly)
