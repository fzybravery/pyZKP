import unittest

from crypto.field.fr import FR_MODULUS
from crypto.poly import omega_for_size
from runtime import CPUContext, Executor, KernelRegistry
from runtime.ir import Device, DType, Graph, OpType
from runtime.kernels.cpu import register_cpu_kernels
from runtime.memory import CPUMemoryPool


class TestRuntimeContextPool(unittest.TestCase):
    def test_pool_reuse_with_context(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg)
        exe = Executor(registry=reg)
        pool = CPUMemoryPool()
        ctx = CPUContext(pool=pool)

        n = 8
        omega = omega_for_size(n)
        shift = 7
        coeffs = [i % FR_MODULUS for i in range(n)]

        def run_once():
            g = Graph()
            g.add_buffer(id="coeff", device=Device.CPU, dtype=DType.FR, data=list(coeffs))
            g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["coeff"], outputs=["ev"], attrs={"n": n, "omega": omega, "shift": shift})
            g.add_node(op=OpType.COSET_COEFFS_FROM_EVALS, inputs=["ev"], outputs=["coeff2"], attrs={"omega": omega, "shift": shift})
            exe.run(g, keep=["coeff2"], context=ctx)
            self.assertEqual(len(g.buffers["coeff2"].data), n)

        run_once()
        reuse0 = pool.cpu_stats.reuse_calls
        run_once()
        self.assertGreater(pool.cpu_stats.reuse_calls, reuse0)


if __name__ == "__main__":
    unittest.main()

