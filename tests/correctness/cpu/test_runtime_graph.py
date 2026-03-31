import unittest

from pyZKP.runtime import Executor, KernelRegistry
from pyZKP.runtime.ir import Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels


class TestRuntimeGraph(unittest.TestCase):
    def test_topo_sort_allows_out_of_order_nodes(self):
        # 故意打乱节点顺序：先消费 c 再生成 c，Topo 排序后仍应能执行成功。
        reg = KernelRegistry()
        register_cpu_kernels(reg)
        exe = Executor(registry=reg)

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=[2, 3, 4, 5])
        g.add_buffer(id="b", device=Device.CPU, dtype=DType.FR, data=[7, 11, 13, 17])

        g.add_node(op=OpType.BATCH_INV, inputs=["c"], outputs=["d"])
        g.add_node(op=OpType.POINTWISE_MUL, inputs=["a", "b"], outputs=["c"])

        exe.run(g, keep=["d"])
        d = g.buffers["d"].data
        self.assertEqual(len(d), 4)


if __name__ == "__main__":
    unittest.main()
