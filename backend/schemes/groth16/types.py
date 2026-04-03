from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from common.crypto.ecc.bn254 import G1, G2


@dataclass(frozen=True)
class Proof:
    a: G1
    b: G2
    c: G1


@dataclass(frozen=True)
class VerifyingKey:
    alpha_g1: G1
    beta_g2: G2
    gamma_g2: G2
    delta_g2: G2
    ic: List[G1]
    public_names: List[str]


@dataclass(frozen=True)
class ProvingKey:
    vk: VerifyingKey
    beta_g1: G1
    delta_g1: G1
    a_query: List[G1]
    b_query_g2: List[G2]
    b_query_g1: List[G1]
    h_query: List[G1]
    l_query: List[G1]
    aux_ids: List[int]
    trapdoor_tau: int
    trapdoor_alpha: int
    trapdoor_beta: int
    trapdoor_gamma: int
    trapdoor_delta: int
