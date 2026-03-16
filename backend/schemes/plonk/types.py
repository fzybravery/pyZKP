from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from pyZKP.common.crypto.ecc.bn254 import G1


@dataclass(frozen=True)
class Domain:
    n: int
    omega: int
    roots: Tuple[int, ...]


@dataclass(frozen=True)
class Gate:
    l: int
    r: int
    o: int
    ql: int
    qr: int
    qm: int
    qo: int
    qc: int
    kind: str


@dataclass(frozen=True)
class Circuit:
    domain: Domain
    one_id: int
    public_var_ids: List[int]
    public_names: List[str]
    gates: Tuple[Gate, ...]
    k1: int
    k2: int
    sigma1_eval: Tuple[int, ...]
    sigma2_eval: Tuple[int, ...]
    sigma3_eval: Tuple[int, ...]
    ql_eval: Tuple[int, ...]
    qr_eval: Tuple[int, ...]
    qm_eval: Tuple[int, ...]
    qo_eval: Tuple[int, ...]
    qc_eval: Tuple[int, ...]


@dataclass(frozen=True)
class VerifyingKey:
    domain: Domain
    one_id: int
    public_names: List[str]
    k1: int
    k2: int
    srs_tau_g2: object
    srs_g2: object
    cm_sigma1: G1
    cm_sigma2: G1
    cm_sigma3: G1
    cm_ql: G1
    cm_qr: G1
    cm_qm: G1
    cm_qo: G1
    cm_qc: G1


@dataclass(frozen=True)
class ProvingKey:
    circuit: Circuit
    vk: VerifyingKey
    srs: object
    coeff_sigma1: Tuple[int, ...]
    coeff_sigma2: Tuple[int, ...]
    coeff_sigma3: Tuple[int, ...]
    coeff_ql: Tuple[int, ...]
    coeff_qr: Tuple[int, ...]
    coeff_qm: Tuple[int, ...]
    coeff_qo: Tuple[int, ...]
    coeff_qc: Tuple[int, ...]


@dataclass(frozen=True)
class Proof:
    cm_a: G1
    cm_b: G1
    cm_c: G1
    cm_z: G1
    cm_t1: G1
    cm_t2: G1
    cm_t3: G1
    evals_zeta: Dict[str, int]
    eval_zeta_omega: int
    pi_zeta: G1
    pi_zeta_omega: G1
