import unittest

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.runtime import Executor, KernelRegistry, RuntimeConfig
from pyZKP.runtime.ir import Backend, Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels
from pyZKP.runtime.kernels.metal import register_metal_kernels
from pyZKP.runtime.metal import metal_available


@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestRuntimeMetalFRPointwise(unittest.TestCase):
    def test_pointwise_mul_matches_cpu(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        a = [1, 2, 3, 1234567890123456789, FR_MODULUS - 2]
        b = [5, 7, 11, 9876543210987654321, FR_MODULUS - 3]
        exp = [(int(a[i]) * int(b[i])) % int(FR_MODULUS) for i in range(len(a))]

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=list(a))
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=list(b))
        g.add_node(op=OpType.TO_DEVICE, inputs=["a"], outputs=["a_m"])
        g.add_node(op=OpType.TO_DEVICE, inputs=["b"], outputs=["b_m"])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a_m", "b_m"], outputs=["c_m"])
        g.add_node(op=OpType.FROM_DEVICE, inputs=["c_m"], outputs=["c"])

        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["c"])
        self.assertEqual(g.buffers["c"].data, exp)


if __name__ == "__main__":
    unittest.main()

