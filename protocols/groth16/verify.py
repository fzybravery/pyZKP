from __future__ import annotations

from typing import List, Sequence

from protocols.groth16.types import Proof, VerifyingKey
from crypto.ecc.bn254 import G1, G1_ZERO, g1_add, g1_mul
from crypto.pairing import pairing_g1_g2
from crypto.field.fr import FR_MODULUS


def verify(vk: VerifyingKey, public_values: Sequence[int], proof: Proof) -> bool:
    if len(public_values) != len(vk.ic):
        return False
    scalars = [int(x) % FR_MODULUS for x in public_values]
    vk_x: G1 = G1_ZERO
    for s, p in zip(scalars, vk.ic):
        vk_x = g1_add(vk_x, g1_mul(p, s))

    left = pairing_g1_g2(proof.a, proof.b)
    right = pairing_g1_g2(vk.alpha_g1, vk.beta_g2)
    right *= pairing_g1_g2(vk_x, vk.gamma_g2)
    right *= pairing_g1_g2(proof.c, vk.delta_g2)
    return left == right
