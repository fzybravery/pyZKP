import unittest

from pyZKP.runtime import Executor, KernelRegistry, RuntimeConfig
from pyZKP.runtime.ir import Backend, Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels
from pyZKP.runtime.kernels.metal import register_metal_kernels
from pyZKP.runtime.metal import metal_available
from pyZKP.runtime.trace import Trace


@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestRuntimeMetalMix(unittest.TestCase):
    def test_cpu_to_metal_pointwise_to_cpu(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=[2, 3, 4])
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=[5, 7, 11])
        g.add_node(op=OpType.TO_DEVICE, inputs=["a"], outputs=["a_m"])
        g.add_node(op=OpType.TO_DEVICE, inputs=["b"], outputs=["b_m"])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a_m", "b_m"], outputs=["c_m"])
        g.add_node(op=OpType.FROM_DEVICE, inputs=["c_m"], outputs=["c"])

        trace = Trace()
        exe.run(g, trace=trace, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["c"])
        self.assertEqual(g.buffers["c"].data, [10, 21, 44])
        self.assertEqual(trace.events[-1].backend, Backend.METAL)

    def test_backend_from_runtime_config(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=[9, 10])
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=[2, 3])
        g.add_node(op=OpType.TO_DEVICE, inputs=["a"], outputs=["a_m"])
        g.add_node(op=OpType.TO_DEVICE, inputs=["b"], outputs=["b_m"])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a_m", "b_m"], outputs=["c_m"])
        g.add_node(op=OpType.FROM_DEVICE, inputs=["c_m"], outputs=["c"])

        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["c"])
        self.assertEqual(g.buffers["c"].data, [18, 30])


if __name__ == "__main__":
    unittest.main()
