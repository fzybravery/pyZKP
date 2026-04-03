import unittest

from common.crypto.field.fr import FR_MODULUS
from runtime import Executor, KernelRegistry, RuntimeConfig
from runtime.ir import Backend, Device, DType, Graph, OpType
from runtime.kernels.cpu import register_cpu_kernels
from runtime.kernels.metal import register_metal_kernels
from runtime.metal import metal_available


@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestRuntimeMetalFRPolySub(unittest.TestCase):
    def test_poly_sub_matches_cpu(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        n = 100
        a_data = [(i * 123 + 456) % int(FR_MODULUS) for i in range(n)]
        b_data = [(i * 789 + 123) % int(FR_MODULUS) for i in range(n)]

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=list(a_data))
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=list(b_data))
        g.add_node(op=OpType.TO_DEVICE, inputs=["a"], outputs=["a_m"])
        g.add_node(op=OpType.TO_DEVICE, inputs=["b"], outputs=["b_m"])
        g.add_node(op=OpType.POLY_SUB, inputs=["a_m", "b_m"], outputs=["c_m"])
        g.add_node(op=OpType.FROM_DEVICE, inputs=["c_m"], outputs=["c"])
        
        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["c"])

        exp = [(a - b) % int(FR_MODULUS) for a, b in zip(a_data, b_data)]
        self.assertEqual(g.buffers["c"].data, exp)

if __name__ == "__main__":
    unittest.main()
