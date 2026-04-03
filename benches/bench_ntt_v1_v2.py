import unittest
import time
import random
from crypto.field.fr import FR_MODULUS
from crypto.poly.ntt import omega_for_size, evals_from_coeffs_on_roots, coeffs_from_evals_on_roots
from runtime import Executor, KernelRegistry, RuntimeConfig
from runtime.ir import Backend, Device, DType, Graph, OpType
from runtime.kernels.cpu import register_cpu_kernels
from runtime.kernels.metal import register_metal_kernels
from runtime.metal import metal_available

@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestNTTPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 4194304
        cls.omega = omega_for_size(cls.n)
        cls.coeffs = [random.randint(0, FR_MODULUS - 1) for _ in range(cls.n)]
        
        cls.reg = KernelRegistry()
        register_cpu_kernels(cls.reg, backend=Backend.METAL)
        register_metal_kernels(cls.reg)
        cls.exe = Executor(registry=cls.reg)

    def _build_ntt_graph(self):
        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=list(self.coeffs))
        g.add_node(op=OpType.TO_DEVICE, inputs=["a"], outputs=["a_m"])
        g.add_node(op=OpType.ROOTS_EVALS_FROM_COEFFS, inputs=["a_m"], outputs=["ev_m"], attrs={"n": self.n, "omega": self.omega})
        g.add_node(op=OpType.FROM_DEVICE, inputs=["ev_m"], outputs=["ev"])
        return g

    def _build_intt_graph(self, evals):
        g = Graph()
        g.add_buffer(id="ev", device=Device.CPU, dtype=DType.FR, data=list(evals))
        g.add_node(op=OpType.TO_DEVICE, inputs=["ev"], outputs=["ev_m"])
        g.add_node(op=OpType.ROOTS_COEFFS_FROM_EVALS, inputs=["ev_m"], outputs=["coeff_m"], attrs={"omega": self.omega})
        g.add_node(op=OpType.FROM_DEVICE, inputs=["coeff_m"], outputs=["coeff"])
        return g

    def test_ntt_correctness_and_perf(self):
        print(f"\n[NTT] n={self.n}")
        # CPU
        t0 = time.time()
        evals_cpu = evals_from_coeffs_on_roots(self.coeffs, n=self.n, omega=self.omega)
        t1 = time.time()
        cpu_time = t1 - t0
        print(f"[NTT] CPU 时间: {cpu_time:.4f} 秒")

        # Metal V1
        g_v1 = self._build_ntt_graph()
        self.exe.run(g_v1, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v1"), keep=["ev"])
        
        g_v1_perf = self._build_ntt_graph()
        t0 = time.time()
        self.exe.run(g_v1_perf, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v1"), keep=["ev"])
        t1 = time.time()
        v1_time = t1 - t0
        print(f"[NTT] Metal V1 时间: {v1_time:.4f} 秒")
        evals_v1 = g_v1_perf.buffers["ev"].data

        # Metal V2
        g_v2 = self._build_ntt_graph()
        self.exe.run(g_v2, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v2"), keep=["ev"])

        g_v2_perf = self._build_ntt_graph()
        t0 = time.time()
        self.exe.run(g_v2_perf, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v2"), keep=["ev"])
        t1 = time.time()
        v2_time = t1 - t0
        print(f"[NTT] Metal V2 时间: {v2_time:.4f} 秒")
        evals_v2 = g_v2_perf.buffers["ev"].data

        # 验证
        for i in range(self.n):
            self.assertEqual(evals_v1[i], evals_cpu[i], f"V1 结果不匹配: index {i}")
            self.assertEqual(evals_v2[i], evals_cpu[i], f"V2 结果不匹配: index {i}")

        print(f"[NTT] V1 vs CPU 加速比: {cpu_time / v1_time:.2f}x")
        print(f"[NTT] V2 vs CPU 加速比: {cpu_time / v2_time:.2f}x")
        print(f"[NTT] V2 vs V1 加速比: {v1_time / v2_time:.2f}x")

    def test_intt_correctness_and_perf(self):
        print(f"\n[INTT] n={self.n}")
        evals_cpu = evals_from_coeffs_on_roots(self.coeffs, n=self.n, omega=self.omega)
        
        # CPU
        t0 = time.time()
        coeffs_cpu = coeffs_from_evals_on_roots(evals_cpu, omega=self.omega)
        t1 = time.time()
        cpu_time = t1 - t0
        print(f"[INTT] CPU 时间: {cpu_time:.4f} 秒")

        # Metal V1
        g_v1 = self._build_intt_graph(evals_cpu)
        self.exe.run(g_v1, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v1"), keep=["coeff"])

        g_v1_perf = self._build_intt_graph(evals_cpu)
        t0 = time.time()
        self.exe.run(g_v1_perf, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v1"), keep=["coeff"])
        t1 = time.time()
        v1_time = t1 - t0
        print(f"[INTT] Metal V1 时间: {v1_time:.4f} 秒")
        coeffs_v1 = g_v1_perf.buffers["coeff"].data

        # Metal V2
        g_v2 = self._build_intt_graph(evals_cpu)
        self.exe.run(g_v2, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v2"), keep=["coeff"])

        g_v2_perf = self._build_intt_graph(evals_cpu)
        t0 = time.time()
        self.exe.run(g_v2_perf, runtime_config=RuntimeConfig(backend=Backend.METAL, metal_ntt_mode="v2"), keep=["coeff"])
        t1 = time.time()
        v2_time = t1 - t0
        print(f"[INTT] Metal V2 时间: {v2_time:.4f} 秒")
        coeffs_v2 = g_v2_perf.buffers["coeff"].data

        # 验证
        for i in range(self.n):
            self.assertEqual(coeffs_v1[i], coeffs_cpu[i], f"INTT V1 结果不匹配: index {i}")
            self.assertEqual(coeffs_v2[i], coeffs_cpu[i], f"INTT V2 结果不匹配: index {i}")

        print(f"[INTT] V1 vs CPU 加速比: {cpu_time / v1_time:.2f}x")
        print(f"[INTT] V2 vs CPU 加速比: {cpu_time / v2_time:.2f}x")
        print(f"[INTT] V2 vs V1 加速比: {v1_time / v2_time:.2f}x")

if __name__ == '__main__':
    unittest.main()
