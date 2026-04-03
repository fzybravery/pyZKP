import unittest
import time
import array
import os

from crypto.ecc.bn254 import G1_GENERATOR, g1_eq
from crypto.field.fr import FR_MODULUS
from crypto.msm.pippenger import msm_pippenger
from runtime.metal.runtime import MetalRuntime, metal_available
from runtime.metal.msm import metal_msm_g1, metal_msm_g1_v2

class TestMSMPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not metal_available():
            raise unittest.SkipTest("Metal runtime not available")
        cls.rt = MetalRuntime.create_default()
        
        cls.n = 16384 # 使用 8K 规模进行单测，既能看出性能差异，又不会导致 CI 超时
        print(f"\n[Setup] Generating {cls.n} points and scalars for MSM Performance Test...")
        
        # 1. 生成点 (为了快，直接复用 G1_GENERATOR)
        cls.points = [G1_GENERATOR for _ in range(cls.n)]
        
        # 2. 生成随机标量
        import random
        random.seed(42) # 固定种子保证每次跑的数据一样
        cls.scalars_int = [random.randint(0, FR_MODULUS - 1) for _ in range(cls.n)]
        
        # 3. 转换标量到 Montgomery 形式并打包进 bytearray (供 Metal V1 和 V2 读取)
        _FR_P = FR_MODULUS
        _FR_R = pow(2, 256, _FR_P)
        _FR_MASK64 = (1 << 64) - 1
        
        flat = []
        for x in cls.scalars_int:
            xm = (x * _FR_R) % _FR_P
            flat.append(int(xm & _FR_MASK64))
            flat.append(int((xm >> 64) & _FR_MASK64))
            flat.append(int((xm >> 128) & _FR_MASK64))
            flat.append(int((xm >> 192) & _FR_MASK64))
            
        arr = array.array("Q", flat)
        cls.scalars_bytes = arr.tobytes()
        cls.mtl_buffer = cls.rt.device.newBufferWithBytes_length_options_(cls.scalars_bytes, len(cls.scalars_bytes), 0)
        
        # Warmup GPU
        _ = metal_msm_g1(cls.rt, cls.points[:100], cls.mtl_buffer, 100)
        _ = metal_msm_g1_v2(cls.rt, cls.points[:100], cls.mtl_buffer, 100)

    def test_msm_correctness_and_perf(self):
        print(f"\n--- Running MSM Performance Comparison (N={self.n}) ---")
        
        # 1. CPU (Pippenger)
        t0 = time.time()
        res_cpu = msm_pippenger(self.points, self.scalars_int)
        time_cpu = time.time() - t0
        print(f"CPU (Pippenger) Time: {time_cpu:.4f} s")
        
        # 2. Metal V1 (Baseline Bucket-centric)
        t0 = time.time()
        res_v1 = metal_msm_g1(self.rt, self.points, self.mtl_buffer, self.n)
        time_v1 = time.time() - t0
        print(f"Metal V1 Time       : {time_v1:.4f} s")
        
        # 3. Metal V2 (Signed-Digit + CSR)
        t0 = time.time()
        res_v2 = metal_msm_g1_v2(self.rt, self.points, self.mtl_buffer, self.n)
        time_v2 = time.time() - t0
        print(f"Metal V2 Time       : {time_v2:.4f} s")
        
        # --- Correctness Check ---
        self.assertTrue(g1_eq(res_cpu, res_v1), "Metal V1 result does not match CPU")
        self.assertTrue(g1_eq(res_v1, res_v2), "Metal V2 result does not match Metal V1")
        print("✅ Correctness Verified: CPU == Metal_V1 == Metal_V2")
        
        # --- Speedup Report ---
        speedup_v1_vs_cpu = time_cpu / time_v1 if time_v1 > 0 else 0
        speedup_v2_vs_v1 = time_v1 / time_v2 if time_v2 > 0 else 0
        speedup_v2_vs_cpu = time_cpu / time_v2 if time_v2 > 0 else 0
        
        print(f"\n--- Speedup Report ---")
        print(f"Metal V1 vs CPU : {speedup_v1_vs_cpu:.2f}x")
        print(f"Metal V2 vs V1  : {speedup_v2_vs_v1:.2f}x  <-- (CSR + Signed-Digit Gain)")
        print(f"Metal V2 vs CPU : {speedup_v2_vs_cpu:.2f}x  <-- (Total GPU Gain)")
        print("------------------------------------------------------\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)