from __future__ import annotations

from enum import Enum

# 算子类型
class OpType(str, Enum):
    TO_DEVICE = "to_device" # 将数据从 CPU 复制到指定设备
    FROM_DEVICE = "from_device" # 将数据从指定设备复制到 CPU
    ROOTS_EVALS_FROM_COEFFS = "roots_evals_from_coeffs" # 单位根域上，系数到值的映射
    ROOTS_COEFFS_FROM_EVALS = "roots_coeffs_from_evals" # 单位根域上，值到系数的映射
    COSET_EVALS_FROM_COEFFS = "coset_evals_from_coeffs" # 陪集上，系数到值的映射
    COSET_COEFFS_FROM_EVALS = "coset_coeffs_from_evals" # 陪集上，值到系数的映射
    BATCH_INV = "batch_inv" # 批量逆元
    POINTWISE_MUL = "pointwise_mul" # 点值域逐点乘
    POLY_MUL_NTT = "poly_mul_ntt" # 利用 NTT 实现多项式乘法
    POLY_SUB = "poly_sub" # 多项式减法
    DIV_XN_MINUS_1 = "div_xn_minus_1" # 多项式除以 x^n - 1
    PLONK_T_QUOTIENT_EVALS = "plonk_t_quotient_evals" # PLONK quotient 的“点值域融合算子”
    MSM_G1 = "msm_g1"
    MSM_G2 = "msm_g2"
    MSM_G1_BATCH = "msm_g1_batch"
    KZG_COMMIT = "kzg_commit"
    KZG_OPEN = "kzg_open"
    KZG_OPEN_PREP_BATCH = "kzg_open_prep_batch"
    KZG_BATCH_COMMIT = "kzg_batch_commit"
    KZG_BATCH_OPEN = "kzg_batch_open"
