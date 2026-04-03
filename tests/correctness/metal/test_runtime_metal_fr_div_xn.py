import os
import unittest
from runtime.ir.ops import OpType
from runtime.ir.types import Backend, Device, DType
from runtime.ir.graph import Graph
from runtime.kernels.registry import KernelRegistry
from runtime.executor import Executor
from runtime.context import CPUContext, MetalContext
from runtime.memory import CPUMemoryPool
from common.crypto.field.fr import FR_MODULUS
from runtime.metal.runtime import metal_available

class TestRuntimeMetalFRDivXN(unittest.TestCase):
    def setUp(self):
        if not metal_available():
            self.skipTest("Metal is not available on this machine.")

    def test_div_xn_matches_cpu(self):
        from runtime.kernels.cpu import register_cpu_kernels
        from runtime.kernels.metal import register_metal_kernels

        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)
        pool = CPUMemoryPool()
        ctx = MetalContext(pool=pool)

        n = 128
        a_data = [(i * 13 + 7) % FR_MODULUS for i in range(n*2 + 10)]

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=a_data)
        
        g.add_node(op=OpType.DIV_XN_MINUS_1, inputs=["a"], outputs=["q", "r"], attrs={"n": n})

        exe.run(g, context=ctx)
        
        out_q = g.buffers["q"].data
        out_r = g.buffers["r"].data
        
        from common.crypto.poly.fast import poly_div_by_xn_minus_1
        ref_q, ref_r = poly_div_by_xn_minus_1(a_data, n)

        self.assertEqual(len(out_q), len(ref_q))
        for i in range(len(ref_q)):
            self.assertEqual(out_q[i], ref_q[i], f"Q Mismatch at {i}: metal={out_q[i]}, ref={ref_q[i]}")

        self.assertEqual(len(out_r), len(ref_r))
        for i in range(len(ref_r)):
            self.assertEqual(out_r[i], ref_r[i], f"R Mismatch at {i}: metal={out_r[i]}, ref={ref_r[i]}")

if __name__ == "__main__":
    unittest.main()
