import unittest

from pyZKP.runtime import Executor, KernelRegistry, RuntimeConfig
from pyZKP.runtime.context import CPUContext
from pyZKP.runtime.ir import Backend, Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels
from pyZKP.runtime.trace import Trace


class TestRuntimeBackendSwitch(unittest.TestCase):
    def test_backend_selected_by_context(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        exe = Executor(registry=reg)
        ctx = CPUContext(backend=Backend.METAL)

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=[2, 3])
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=[5, 7])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a", "b"], outputs=["c"])

        trace = Trace()
        exe.run(g, trace=trace, context=ctx)
        self.assertEqual(g.buffers["c"].data, [10, 21])
        self.assertEqual(trace.events[-1].backend, Backend.METAL)

    def test_backend_selected_by_runtime_config(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        exe = Executor(registry=reg)

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=[2, 3])
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=[5, 7])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a", "b"], outputs=["c"])

        trace = Trace()
        exe.run(g, trace=trace, runtime_config=RuntimeConfig(backend=Backend.METAL))
        self.assertEqual(g.buffers["c"].data, [10, 21])
        self.assertEqual(trace.events[-1].backend, Backend.METAL)


if __name__ == "__main__":
    unittest.main()

