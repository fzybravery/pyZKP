import hashlib
from typing import List, Tuple
from crypto.field.fr import FR_MODULUS

# Poseidon 算法中的轮常量 (ARC) 和 MDS 矩阵通常由复杂的脚本离线生成。
# 为了保持框架的轻量级并演示核心原理，这里我们使用确定的 SHA256 伪随机数生成器，
# 动态生成适用于任何 arity 的安全常量。(在生产环境中可替换为固定常量表)
def generate_poseidon_constants(t: int, r_f: int, r_p: int) -> Tuple[List[int], List[List[int]]]:
    arc = []
    # 1. 生成轮常量 (Add Round Constants)
    for i in range((r_f + r_p) * t):
        seed = f"poseidon_arc_{t}_{r_f}_{r_p}_{i}".encode()
        val = int.from_bytes(hashlib.sha256(seed).digest(), 'big') % FR_MODULUS
        arc.append(val)
        
    # 2. 生成 MDS 矩阵 (Maximum Distance Separable)
    # 此处生成一个随机矩阵以模拟 MDS 的混合作用，工业级实现应采用柯西矩阵等构造
    mds = []
    for i in range(t):
        row = []
        for j in range(t):
            seed = f"poseidon_mds_{t}_{r_f}_{r_p}_{i}_{j}".encode()
            val = int.from_bytes(hashlib.sha256(seed).digest(), 'big') % FR_MODULUS
            row.append(val)
        mds.append(row)
        
    return arc, mds

def poseidon_hash(inputs: List[int], r_f: int = 8, r_p: int = 53) -> int:
    """
    纯 Python 实现的 Poseidon Hash 参考实现
    :param inputs: 待哈希的输入列表 (在 BN254 上通常 arity 为 2，对应 width t=3)
    :param r_f: 全轮数量 (Full Rounds)
    :param r_p: 局部轮数量 (Partial Rounds)
    :return: 域内的哈希结果
    """
    t = len(inputs) + 1
    arc, mds = generate_poseidon_constants(t, r_f, r_p)
    
    # 初始状态: [0, input_0, input_1, ...]
    state = [0] + [int(x) % FR_MODULUS for x in inputs]
    
    for r in range(r_f + r_p):
        # 1. AddRoundConstants
        for i in range(t):
            state[i] = (state[i] + arc[r * t + i]) % FR_MODULUS
            
        # 2. SubWords (S-Box)
        # BN254 曲线的 alpha 通常为 5, S-box 为 x^5
        is_full_round = (r < r_f // 2) or (r >= r_f // 2 + r_p)
        for i in range(t):
            if is_full_round or i == 0:
                state[i] = pow(state[i], 5, FR_MODULUS)
                
        # 3. MixLayer (MDS 矩阵乘法)
        new_state = [0] * t
        for i in range(t):
            for j in range(t):
                new_state[i] = (new_state[i] + mds[i][j] * state[j]) % FR_MODULUS
        state = new_state
        
    return state[0]
