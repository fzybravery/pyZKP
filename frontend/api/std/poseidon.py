from typing import List
from frontend.api.api import API, Var
from crypto.hash.poseidon import generate_poseidon_constants

def _sbox(api: API, x: Var) -> Var:
    """
    实现 BN254 Poseidon 的 x^5 (S-Box) 操作
    使用最少约束乘法链：x^2, x^4, x^5
    """
    x2 = api.Mul(x, x)
    x4 = api.Mul(x2, x2)
    return api.Mul(x4, x)

def poseidon_circuit(api: API, inputs: List[Var], r_f: int = 8, r_p: int = 53) -> Var:
    """
    在 ZKP 电路中执行 Poseidon 哈希。
    由于没有位运算，只有原生的加法和乘法，Poseidon 可以非常高效地编译为 R1CS 约束。
    
    :param api: 当前构建电路的 API 实例
    :param inputs: 待哈希的变量列表 (通常两个变量组合为一个哈希)
    :param r_f: 全轮数量
    :param r_p: 局部轮数量
    :return: 包含哈希结果的电路变量
    """
    t = len(inputs) + 1
    arc, mds = generate_poseidon_constants(t, r_f, r_p)
    
    # 1. 初始状态 (包含 Capacity = 0)
    state = [api.Constant(0)] + inputs
    
    for r in range(r_f + r_p):
        # AddRoundConstants (ARC): 加法门
        for i in range(t):
            c = api.Constant(arc[r * t + i])
            state[i] = api.Add(state[i], c)
            
        # SubWords (S-Box): 非线性层
        # 全轮时，所有元素均执行 x^5
        # 局部轮时，仅第一个元素执行 x^5
        is_full_round = (r < r_f // 2) or (r >= r_f // 2 + r_p)
        for i in range(t):
            if is_full_round or i == 0:
                state[i] = _sbox(api, state[i])
                
        # MixLayer: 乘法门 + 累加门
        new_state = []
        for i in range(t):
            row_sum = api.Constant(0)
            for j in range(t):
                term = api.Mul(api.Constant(mds[i][j]), state[j])
                row_sum = api.Add(row_sum, term)
            new_state.append(row_sum)
        state = new_state
        
    return state[0]
