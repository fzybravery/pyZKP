import os
import unittest
from pyZKP.runtime.ir.ops import OpType
from pyZKP.runtime.ir.types import Backend, Device, DType
from pyZKP.runtime.ir.graph import Graph
from pyZKP.runtime.kernels.registry import KernelRegistry
from pyZKP.runtime.executor import Executor
from pyZKP.runtime.context import CPUContext, MetalContext
from pyZKP.runtime.memory import CPUMemoryPool
from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.runtime.metal.runtime import metal_available

class TestRuntimeMetalFRBatchInv(unittest.TestCase):
    def setUp(self):
        if not metal_available():
            self.skipTest("Metal is not available on this machine.")

    def test_batch_inv_matches_cpu(self):
        from pyZKP.runtime.kernels.cpu import register_cpu_kernels
        from pyZKP.runtime.kernels.metal import register_metal_kernels

        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)
        pool = CPUMemoryPool()
        ctx = MetalContext(pool=pool)

        n = 1000
        # 构造一些数据，包含 0
        a_data = [(i * 12345 + 6789) % FR_MODULUS for i in range(n)]
        a_data[10] = 0
        a_data[500] = 0

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=a_data)
        
        # 自动图重写会处理 TO_DEVICE / FROM_DEVICE
        g.add_node(op=OpType.BATCH_INV, inputs=["a"], outputs=["inv_a"])

        exe.run(g, context=ctx)
        
        # 检查是否成功执行了 Metal 的 BATCH_INV (如果还没实现，它会回退到 CPU)
        # 这里我们假设之后会实现 Metal 版本的 BATCH_INV，如果没实现，它会自动 fallback，测试依然会通过，只是跑在 CPU 上。
        # 主要是为了验证即使注册了 METAL backend，图也能正确跑通。

        # 拿到结果并和 CPU 直接算出来的比较
        out_metal = g.buffers["inv_a"].data
        
        # 使用 CPU 参考实现
        from pyZKP.common.crypto.field.batch import fr_batch_inv
        ref_inv = fr_batch_inv(a_data)

        self.assertEqual(len(out_metal), len(ref_inv))
        for i in range(n):
            self.assertEqual(out_metal[i], ref_inv[i], f"Mismatch at {i}: metal={out_metal[i]}, ref={ref_inv[i]}")

if __name__ == "__main__":
    unittest.main()
