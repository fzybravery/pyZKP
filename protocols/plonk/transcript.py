from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional

from crypto.field.fr import FR_MODULUS
from crypto.ecc.bn254 import G1
from py_ecc import optimized_bn128 as b


# 整数转换为字节数组
def _i2b(x: int) -> bytes:
    return int(x).to_bytes(32, "big", signed=False)

# 将椭圆曲线点转换为字节数组
def _g1_to_bytes(p: G1) -> bytes:
    x, y = b.normalize(p)[:2]
    return _i2b(int(x.n)) + _i2b(int(y.n))

# Fiat-Shamir 庭审转录本（Transcript）类。
# 核心机制：
# 1. Absorb（吸收）：将证明者生成的公开输入、KZG 承诺等数据单向混淆（Hash）入内部状态。
# 2. Challenge（挑战）：基于当前所有吸收的历史状态，生成伪随机的标量挑战（如 alpha, beta, zeta 等）。
# 3. 链式更新：每次生成挑战后，会将挑战结果重新吸收回状态中，确保前后挑战的强因果绑定（密码学）因果绑定，杜绝证明者作弊。
@dataclass
class Transcript:
    state: "hashlib._Hash"

    def __init__(self, label: bytes = b"pyZKP-plonk") -> None:
        self.state = hashlib.sha256()
        self.state.update(label)

    def absorb_bytes(self, data: bytes) -> None:
        self.state.update(data)

    def absorb_int(self, x: int) -> None:
        self.state.update(_i2b(x % FR_MODULUS))

    def absorb_g1(self, p: G1) -> None:
        self.state.update(_g1_to_bytes(p))

    def challenge(self, label: bytes) -> int:
        h = self.state.copy()
        h.update(label)
        d = h.digest()
        x = int.from_bytes(d, "big") % FR_MODULUS
        self.state.update(d)
        return x
