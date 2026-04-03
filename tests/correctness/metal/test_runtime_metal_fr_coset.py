import unittest

from common.crypto.field.fr import FR_MODULUS
from common.crypto.poly.ntt import omega_for_size, evals_from_coeffs_on_coset, coeffs_from_evals_on_coset
from runtime import Executor, KernelRegistry, RuntimeConfig
from runtime.ir import Backend, Device, DType, Graph, OpType
from runtime.kernels.cpu import register_cpu_kernels
from runtime.kernels.metal import register_metal_kernels
from runtime.metal import metal_available

@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestRuntimeMetalFRCoset(unittest.TestCase):
    def test_coset_evals_matches_cpu(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        n = 16
        omega = omega_for_size(n)
        shift = 5  # Arbitrary shift
        coeff = [i * i % int(FR_MODULUS) for i in range(n)]

        g = Graph()
        g.add_buffer(id="a", device=Device.CPU, dtype=DType.FR, data=list(coeff))
        g.add_node(op=OpType.TO_DEVICE, inputs=["a"], outputs=["a_m"])
        g.add_node(op=OpType.COSET_EVALS_FROM_COEFFS, inputs=["a_m"], outputs=["ev_m"], attrs={"n": n, "omega": omega, "shift": shift})
        g.add_node(op=OpType.FROM_DEVICE, inputs=["ev_m"], outputs=["ev"])
        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["ev"])

        exp = evals_from_coeffs_on_coset(coeff, n=n, omega=omega, shift=shift)
        self.assertEqual(g.buffers["ev"].data, exp)

    def test_coset_coeffs_matches_cpu(self):
        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)

        n = 16
        omega = omega_for_size(n)
        shift = 5
        ev = [(i + 3) * 7 % int(FR_MODULUS) for i in range(n)]

        g = Graph()
        g.add_buffer(id="ev", device=Device.CPU, dtype=DType.FR, data=list(ev))
        g.add_node(op=OpType.TO_DEVICE, inputs=["ev"], outputs=["ev_m"])
        g.add_node(op=OpType.COSET_COEFFS_FROM_EVALS, inputs=["ev_m"], outputs=["coeff_m"], attrs={"omega": omega, "shift": shift})
        g.add_node(op=OpType.FROM_DEVICE, inputs=["coeff_m"], outputs=["coeff"])
        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["coeff"])

        exp = coeffs_from_evals_on_coset(ev, omega=omega, shift=shift)
        self.assertEqual(g.buffers["coeff"].data, exp)

if __name__ == "__main__":
    unittest.main()
