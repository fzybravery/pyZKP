import unittest

from runtime import Executor, KernelRegistry
from runtime.ir import Device, DType, Graph, OpType
from runtime.kernels.cpu import register_cpu_kernels


class TestRuntimeGraphBatch(unittest.TestCase):
    def test_graph_can_run_twice_without_reanalyze(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg)
        exe = Executor(registry=reg)

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=[2, 3, 4, 5])
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=[7, 11, 13, 17])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a", "b"], outputs=["c"])

        exe.run(g)
        self.assertEqual(g.buffers["c"].data, [14, 33, 52, 85])

        exe.run(g)
        self.assertEqual(g.buffers["c"].data, [14, 33, 52, 85])

    def test_graph_analyze_cache_invalidate_on_add_node(self):
        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=[1, 2, 3, 4])
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=[1, 1, 1, 1])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a", "b"], outputs=["c"])
        a1 = g.analyze_cached()
        g.add_node(op=OpType.BATCH_INV, inputs=["c"], outputs=["d"])
        a2 = g.analyze_cached()
        self.assertNotEqual(a1.topo_order, a2.topo_order)


if __name__ == "__main__":
    unittest.main()

