import os
import unittest
from runtime.ir.ops import OpType
from runtime.ir.types import Backend, Device, DType
from runtime.ir.graph import Graph
from runtime.kernels.registry import KernelRegistry
from runtime.executor import Executor
from runtime.context import CPUContext, MetalContext
from runtime.memory import CPUMemoryPool
from crypto.field.fr import FR_MODULUS
from runtime.metal.runtime import metal_available

class TestRuntimeMetalFRPolyMul(unittest.TestCase):
    def setUp(self):
        if not metal_available():
            self.skipTest("Metal is not available on this machine.")

    def test_poly_mul_matches_cpu(self):
        from runtime.kernels.cpu import register_cpu_kernels
        from runtime.kernels.metal import register_metal_kernels

        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)
        pool = CPUMemoryPool()
        ctx = MetalContext(pool=pool)

        n = 128
        a_data = [(i * 13 + 7) % FR_MODULUS for i in range(n)]
        b_data = [(i * 17 + 3) % FR_MODULUS for i in range(n)]

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=a_data)
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=b_data)
        
        g.add_node(op=OpType.POLY_MUL_NTT, inputs=["a", "b"], outputs=["c"])

        exe.run(g, context=ctx)
        
        out_metal = g.buffers["c"].data
        
        from crypto.poly.ntt import poly_mul_ntt
        ref_c = poly_mul_ntt(a_data, b_data)

        self.assertEqual(len(out_metal), len(ref_c))
        for i in range(len(ref_c)):
            self.assertEqual(out_metal[i], ref_c[i], f"Mismatch at {i}: metal={out_metal[i]}, ref={ref_c[i]}")

if __name__ == "__main__":
    unittest.main()
