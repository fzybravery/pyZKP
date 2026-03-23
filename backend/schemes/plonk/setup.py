from __future__ import annotations

from pyZKP.backend.schemes.plonk.lowering import lower_to_circuit
from pyZKP.backend.schemes.plonk.types import ProvingKey, VerifyingKey
from pyZKP.common.crypto.kzg.cpu_ref import commit, setup_srs
from pyZKP.common.crypto.poly import coeffs_from_evals_on_roots
from pyZKP.common.ir.core import CircuitIR


def setup(ir: CircuitIR) -> ProvingKey:
    circuit = lower_to_circuit(ir)
    n = circuit.domain.n
    max_deg = 8 * n
    srs = setup_srs(max_deg)

    omega = circuit.domain.omega
    coeff_sigma1 = tuple(coeffs_from_evals_on_roots(circuit.sigma1_eval, omega=omega))
    coeff_sigma2 = tuple(coeffs_from_evals_on_roots(circuit.sigma2_eval, omega=omega))
    coeff_sigma3 = tuple(coeffs_from_evals_on_roots(circuit.sigma3_eval, omega=omega))
    coeff_ql = tuple(coeffs_from_evals_on_roots(circuit.ql_eval, omega=omega))
    coeff_qr = tuple(coeffs_from_evals_on_roots(circuit.qr_eval, omega=omega))
    coeff_qm = tuple(coeffs_from_evals_on_roots(circuit.qm_eval, omega=omega))
    coeff_qo = tuple(coeffs_from_evals_on_roots(circuit.qo_eval, omega=omega))
    coeff_qc = tuple(coeffs_from_evals_on_roots(circuit.qc_eval, omega=omega))

    cm_sigma1 = commit(srs, coeff_sigma1)
    cm_sigma2 = commit(srs, coeff_sigma2)
    cm_sigma3 = commit(srs, coeff_sigma3)
    cm_ql = commit(srs, coeff_ql)
    cm_qr = commit(srs, coeff_qr)
    cm_qm = commit(srs, coeff_qm)
    cm_qo = commit(srs, coeff_qo)
    cm_qc = commit(srs, coeff_qc)

    vk = VerifyingKey(
        domain=circuit.domain,
        one_id=circuit.one_id,
        public_names=["ONE"] + circuit.public_names,
        k1=circuit.k1,
        k2=circuit.k2,
        srs_tau_g2=srs.g2_powers[1],
        srs_g2=srs.g2_powers[0],
        cm_sigma1=cm_sigma1,
        cm_sigma2=cm_sigma2,
        cm_sigma3=cm_sigma3,
        cm_ql=cm_ql,
        cm_qr=cm_qr,
        cm_qm=cm_qm,
        cm_qo=cm_qo,
        cm_qc=cm_qc,
    )

    return ProvingKey(
        circuit=circuit,
        vk=vk,
        srs=srs,
        coeff_sigma1=coeff_sigma1,
        coeff_sigma2=coeff_sigma2,
        coeff_sigma3=coeff_sigma3,
        coeff_ql=coeff_ql,
        coeff_qr=coeff_qr,
        coeff_qm=coeff_qm,
        coeff_qo=coeff_qo,
        coeff_qc=coeff_qc,
    )
