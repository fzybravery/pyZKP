import unittest

from common.crypto.field.fr import FR_MODULUS
from common.crypto.poly.ntt import omega_for_size
from runtime import Executor, KernelRegistry, RuntimeConfig
from runtime.ir import Backend, Device, DType, Graph, OpType
from runtime.kernels.cpu import register_cpu_kernels
from runtime.kernels.metal import register_metal_kernels
from runtime.metal import metal_available


@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestRuntimeMetalFRNTT(unittest.TestCase):
    def test_roots_ntt_matches_cpu(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        n = 16
        omega = omega_for_size(n)
        coeff = [i * i % int(FR_MODULUS) for i in range(n)]

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=list(coeff))
        g.add_node(op=OpType.TO_DEVICE, inputs=["a"], outputs=["a_m"])
        g.add_node(op=OpType.ROOTS_EVALS_FROM_COEFFS, inputs=["a_m"], outputs=["ev_m"], attrs={"n": n, "omega": omega})
        g.add_node(op=OpType.FROM_DEVICE, inputs=["ev_m"], outputs=["ev"])
        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["ev"])

        from common.crypto.poly.ntt import evals_from_coeffs_on_roots

        exp = evals_from_coeffs_on_roots(coeff, n=n, omega=omega)
        self.assertEqual(g.buffers["ev"].data, exp)

    def test_roots_intt_roundtrip(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        n = 16
        omega = omega_for_size(n)
        ev = [(i + 3) * 7 % int(FR_MODULUS) for i in range(n)]

        g = Graph()
        g.add_buffer(id="ev", device=Device.CPU, dtype=DType.FR, data=list(ev))
        g.add_node(op=OpType.TO_DEVICE, inputs=["ev"], outputs=["ev_m"])
        g.add_node(op=OpType.ROOTS_COEFFS_FROM_EVALS, inputs=["ev_m"], outputs=["coeff_m"], attrs={"omega": omega})
        g.add_node(op=OpType.FROM_DEVICE, inputs=["coeff_m"], outputs=["coeff"])
        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["coeff"])

        from common.crypto.poly.ntt import coeffs_from_evals_on_roots

        exp = coeffs_from_evals_on_roots(ev, omega=omega)
        self.assertEqual(g.buffers["coeff"].data, exp)


if __name__ == "__main__":
    unittest.main()

