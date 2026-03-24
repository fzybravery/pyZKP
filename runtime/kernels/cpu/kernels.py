from __future__ import annotations

from typing import Any, Dict, List

from pyZKP.common.crypto.field import FR_MODULUS, fr_batch_inv
from pyZKP.common.crypto.kzg.cpu_ref import commit as kzg_commit, open_proof as kzg_open
from pyZKP.common.crypto.msm import msm_naive_g1, msm_naive_g2
from pyZKP.common.crypto.poly import (
    coeffs_from_evals_on_coset,
    coeffs_from_evals_on_roots,
    evals_from_coeffs_on_coset,
    evals_from_coeffs_on_roots,
    poly_div_by_xn_minus_1,
    poly_mul_ntt,
)
from pyZKP.common.crypto.poly.cpu_ref import poly_sub
from pyZKP.runtime.ir.ops import OpType
from pyZKP.runtime.ir.types import Buffer, Device, DType
from pyZKP.runtime.kernels.registry import KernelRegistry


# cpu算子注册入口
def register_cpu_kernels(registry: KernelRegistry) -> None:
    registry.register(OpType.ROOTS_EVALS_FROM_COEFFS, Device.CPU, _roots_evals_from_coeffs)
    registry.register(OpType.ROOTS_COEFFS_FROM_EVALS, Device.CPU, _roots_coeffs_from_evals)
    registry.register(OpType.COSET_EVALS_FROM_COEFFS, Device.CPU, _coset_evals_from_coeffs)
    registry.register(OpType.COSET_COEFFS_FROM_EVALS, Device.CPU, _coset_coeffs_from_evals)
    registry.register(OpType.BATCH_INV, Device.CPU, _batch_inv)
    registry.register(OpType.POINTWISE_MUL, Device.CPU, _pointwise_mul)
    registry.register(OpType.POLY_MUL_NTT, Device.CPU, _poly_mul_ntt)
    registry.register(OpType.POLY_SUB, Device.CPU, _poly_sub)
    registry.register(OpType.DIV_XN_MINUS_1, Device.CPU, _div_xn_minus_1)
    registry.register(OpType.PLONK_T_QUOTIENT_EVALS, Device.CPU, _plonk_t_quotient_evals)
    registry.register(OpType.MSM_G1, Device.CPU, _msm_g1)
    registry.register(OpType.MSM_G2, Device.CPU, _msm_g2)
    registry.register(OpType.KZG_COMMIT, Device.CPU, _kzg_commit)
    registry.register(OpType.KZG_OPEN, Device.CPU, _kzg_open)


def _roots_evals_from_coeffs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    n = int(ctx["attrs"]["n"])
    omega = int(ctx["attrs"]["omega"])
    out_id = node.outputs[0]
    ev = evals_from_coeffs_on_roots(inp.data, n=n, omega=omega)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=ev, meta={"n": n, "omega": omega})}}


def _roots_coeffs_from_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    omega = int(ctx["attrs"]["omega"])
    out_id = node.outputs[0]
    coeff = coeffs_from_evals_on_roots(inp.data, omega=omega)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=coeff, meta={"omega": omega})}}


def _coset_evals_from_coeffs(ctx: Dict[str, Any]) -> Dict[str, Any]:

    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    n = int(ctx["attrs"]["n"])
    omega = int(ctx["attrs"]["omega"])
    shift = int(ctx["attrs"]["shift"])
    out_id = node.outputs[0]
    ev = evals_from_coeffs_on_coset(inp.data, n=n, omega=omega, shift=shift)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=ev, meta={"n": n, "omega": omega, "shift": shift})}}


def _coset_coeffs_from_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    omega = int(ctx["attrs"]["omega"])
    shift = int(ctx["attrs"]["shift"])
    out_id = node.outputs[0]
    coeff = coeffs_from_evals_on_coset(inp.data, omega=omega, shift=shift)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=coeff, meta={"omega": omega, "shift": shift})}}


def _batch_inv(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    out_id = node.outputs[0]
    inv = fr_batch_inv(inp.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=inp.device, dtype=DType.FR, data=inv, meta=inp.meta)}}


def _pointwise_mul(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    if len(a.data) != len(b.data):
        raise ValueError("length mismatch")
    out = [(int(a.data[i]) * int(b.data[i])) % FR_MODULUS for i in range(len(a.data))]
    return {"outputs": {out_id: Buffer(id=out_id, device=a.device, dtype=DType.FR, data=out)}}


def _poly_mul_ntt(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    out = poly_mul_ntt(a.data, b.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=a.device, dtype=DType.FR, data=out)}}


def _poly_sub(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    a: Buffer = ctx["inputs"][0]
    b: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    out = poly_sub(a.data, b.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=a.device, dtype=DType.FR, data=out)}}


def _div_xn_minus_1(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    inp: Buffer = ctx["inputs"][0]
    n = int(ctx["attrs"]["n"])
    q_id = node.outputs[0]
    r_id = node.outputs[1]
    q, r = poly_div_by_xn_minus_1(inp.data, n)
    return {"outputs": {q_id: Buffer(id=q_id, device=inp.device, dtype=DType.FR, data=q), r_id: Buffer(id=r_id, device=inp.device, dtype=DType.FR, data=r)}}


def _msm_g1(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    points: Buffer = ctx["inputs"][0]
    scalars: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    acc = msm_naive_g1(points.data, scalars.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=points.device, dtype=DType.G1, data=acc)}}


def _msm_g2(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    points: Buffer = ctx["inputs"][0]
    scalars: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    acc = msm_naive_g2(points.data, scalars.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=points.device, dtype=DType.G2, data=acc)}}


def _kzg_commit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    coeffs: Buffer = ctx["inputs"][1]
    out_id = node.outputs[0]
    cm = kzg_commit(srs.data, coeffs.data)
    return {"outputs": {out_id: Buffer(id=out_id, device=coeffs.device, dtype=DType.G1, data=cm)}}


def _kzg_open(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    srs: Buffer = ctx["inputs"][0]
    coeffs: Buffer = ctx["inputs"][1]
    z = int(ctx["attrs"]["z"])
    y_id = node.outputs[0]
    pi_id = node.outputs[1]
    y, pi = kzg_open(srs.data, coeffs.data, z)
    return {
        "outputs": {
            y_id: Buffer(id=y_id, device=coeffs.device, dtype=DType.FR, data=int(y) % FR_MODULUS),
            pi_id: Buffer(id=pi_id, device=coeffs.device, dtype=DType.G1, data=pi),
        }
    }


def _plonk_t_quotient_evals(ctx: Dict[str, Any]) -> Dict[str, Any]:
    node = ctx["node"]
    attrs = ctx["attrs"]
    n = int(attrs["n"])
    m = int(attrs["m"])
    omega_m = int(attrs["omega_m"])
    shift = int(attrs["shift"])
    shift_n = int(attrs["shift_n"])
    alpha = int(attrs["alpha"]) % FR_MODULUS
    beta = int(attrs["beta"]) % FR_MODULUS
    gamma = int(attrs["gamma"]) % FR_MODULUS
    k1 = int(attrs["k1"]) % FR_MODULUS
    k2 = int(attrs["k2"]) % FR_MODULUS
    u = pow(int(omega_m) % FR_MODULUS, n, FR_MODULUS)

    (
        a_ext,
        b_ext,
        c_ext,
        z_ext,
        pi_ext,
        ql_ext,
        qr_ext,
        qm_ext,
        qo_ext,
        qc_ext,
        s1_ext,
        s2_ext,
        s3_ext,
        zshift_ext,
        l1_ext,
    ) = [ctx["inputs"][i].data for i in range(15)]

    if not all(len(v) == m for v in [a_ext, b_ext, c_ext, z_ext, pi_ext, ql_ext, qr_ext, qm_ext, qo_ext, qc_ext, s1_ext, s2_ext, s3_ext, zshift_ext, l1_ext]):
        raise ValueError("eval length mismatch")

    alpha2 = (alpha * alpha) % FR_MODULUS

    x = shift % FR_MODULUS
    u_pow = 1
    zh: List[int] = []
    num: List[int] = []
    for i in range(m):
        a = int(a_ext[i]) % FR_MODULUS
        b = int(b_ext[i]) % FR_MODULUS
        cc = int(c_ext[i]) % FR_MODULUS
        z = int(z_ext[i]) % FR_MODULUS

        gate = (int(ql_ext[i]) * a + int(qr_ext[i]) * b) % FR_MODULUS
        gate = (gate + int(qm_ext[i]) * a % FR_MODULUS * b) % FR_MODULUS
        gate = (gate + int(qo_ext[i]) * cc + int(qc_ext[i]) + int(pi_ext[i])) % FR_MODULUS

        t1 = (a + beta * x + gamma) % FR_MODULUS
        t2 = (b + beta * k1 % FR_MODULUS * x + gamma) % FR_MODULUS
        t3 = (cc + beta * k2 % FR_MODULUS * x + gamma) % FR_MODULUS
        left = (t1 * t2) % FR_MODULUS
        left = (left * t3) % FR_MODULUS
        left = (left * z) % FR_MODULUS

        u1v = (a + beta * int(s1_ext[i]) + gamma) % FR_MODULUS
        u2v = (b + beta * int(s2_ext[i]) + gamma) % FR_MODULUS
        u3v = (cc + beta * int(s3_ext[i]) + gamma) % FR_MODULUS
        right = (u1v * u2v) % FR_MODULUS
        right = (right * u3v) % FR_MODULUS
        right = (right * int(zshift_ext[i])) % FR_MODULUS

        perm = (left - right) % FR_MODULUS
        boundary = ((z - 1) % FR_MODULUS) * (int(l1_ext[i]) % FR_MODULUS) % FR_MODULUS
        num.append((gate + alpha * perm + alpha2 * boundary) % FR_MODULUS)

        x_n = (shift_n * u_pow) % FR_MODULUS
        zh.append((x_n - 1) % FR_MODULUS)

        u_pow = (u_pow * u) % FR_MODULUS
        x = (x * omega_m) % FR_MODULUS

    out_num = node.outputs[0]
    out_zh = node.outputs[1]
    return {
        "outputs": {
            out_num: Buffer(id=out_num, device=Device.CPU, dtype=DType.FR, data=num, meta={"m": m}),
            out_zh: Buffer(id=out_zh, device=Device.CPU, dtype=DType.FR, data=zh, meta={"m": m}),
        }
    }
