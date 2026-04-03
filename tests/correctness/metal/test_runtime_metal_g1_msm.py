import os
import unittest
from runtime.ir.ops import OpType
from runtime.ir.types import Backend, Device, DType
from runtime.ir.graph import Graph
from runtime.kernels.registry import KernelRegistry
from runtime.executor import Executor
from runtime.context import MetalContext
from runtime.memory import CPUMemoryPool
from common.crypto.field.fr import FR_MODULUS
from common.crypto.ecc.bn254 import G1_GENERATOR, g1_mul
from runtime.metal.runtime import metal_available

class TestRuntimeMetalG1MSM(unittest.TestCase):
    def setUp(self):
        if not metal_available():
            self.skipTest("Metal is not available on this machine.")

    def test_msm_g1_matches_cpu(self):
        from runtime.kernels.cpu import register_cpu_kernels
        from runtime.kernels.metal import register_metal_kernels

        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)
        pool = CPUMemoryPool()
        # Ensure we create a valid MetalContext with runtime initialized
        from runtime.config import RuntimeConfig
        rc = RuntimeConfig(backend=Backend.METAL)
        ctx = rc.make_context()

        n = 100
        # 1. 生成 n 个随机标量
        scalars_data = [(i * 12345 + 6789) % FR_MODULUS for i in range(n)]
        
        # 2. 生成 n 个不同的基点 (简单的倍乘即可)
        points_data = [g1_mul(G1_GENERATOR, i + 1) for i in range(n)]

        g = Graph()
        # 基点通常驻留在 CPU (对于 SRS 来说) 或者预先在 GPU，这里我们先放在 CPU，
        # 我们修改过的 metal_msm_g1 会自己把它 pack 后上传。
        g.add_buffer(id="points", device=Device.CPU, dtype=DType.OBJ, data=points_data)
        
        # scalars 在 CPU 上，由于 executor 的 Graph Rewrite Pass，它会自动生成 TO_DEVICE 节点上传到 METAL
        g.add_buffer(id="scalars", device=Device.CPU, dtype=DType.FR, data=scalars_data)
        
        g.add_node(op=OpType.MSM_G1, inputs=["points", "scalars"], outputs=["msm_res"])

        # 运行图
        exe.run(g, context=ctx)
        
        # 拿到 Metal 算出的结果 (它在 Python 封装里被指定为返回到 Device.CPU)
        out_metal = g.buffers["msm_res"].data
        
        # 使用 CPU 参考实现 (Naive 或 Pippenger)
        from common.crypto.msm.pippenger import msm_pippenger
        ref_msm = msm_pippenger(points_data, scalars_data)

        # 验证仿射坐标是否完全一致
        from py_ecc.optimized_bn128 import normalize
        m_norm = normalize(out_metal)
        r_norm = normalize(ref_msm)
        self.assertEqual(m_norm, r_norm, f"MSM mismatch!\nMetal: {m_norm}\nCPU:   {r_norm}")

if __name__ == "__main__":
    unittest.main()
