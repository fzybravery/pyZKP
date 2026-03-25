import secrets
import unittest

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.kzg.cpu_ref import commit, open_proof, setup_srs
from pyZKP.runtime import Executor, KernelRegistry
from pyZKP.runtime.ir import Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels


class TestRuntimeMSMKZG(unittest.TestCase):
    def test_kzg_batch_commit_matches_single(self):
        srs = setup_srs(64)
        polys = []
        for _ in range(5):
            deg = secrets.randbelow(20) + 1
            polys.append([secrets.randbelow(FR_MODULUS) for _ in range(deg)])

        reg = KernelRegistry()
        register_cpu_kernels(reg)
        exe = Executor(registry=reg)
        g = Graph()
        g.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=srs)
        g.add_buffer(id="polys", device=Device.CPU, dtype=DType.OBJ, data=polys)
        g.add_node(op=OpType.KZG_BATCH_COMMIT, inputs=["srs", "polys"], outputs=["cms"])
        exe.run(g, keep=["cms"])
        cms = g.buffers["cms"].data

        expected = [commit(srs, p) for p in polys]
        self.assertEqual(cms, expected)

    def test_kzg_open_matches_reference(self):
        srs = setup_srs(64)
        coeffs = [secrets.randbelow(FR_MODULUS) for _ in range(25)]
        z = secrets.randbelow(FR_MODULUS)

        reg = KernelRegistry()
        register_cpu_kernels(reg)
        exe = Executor(registry=reg)
        g = Graph()
        g.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=srs)
        g.add_buffer(id="coeffs", device=Device.CPU, dtype=DType.FR, data=coeffs)
        g.add_node(op=OpType.KZG_OPEN, inputs=["srs", "coeffs"], outputs=["y", "pi"], attrs={"z": int(z)})
        exe.run(g, keep=["y", "pi"])

        y0, pi0 = open_proof(srs, coeffs, z)
        self.assertEqual(int(g.buffers["y"].data) % FR_MODULUS, int(y0) % FR_MODULUS)
        self.assertEqual(g.buffers["pi"].data, pi0)

    def test_kzg_batch_open_matches_reference(self):
        srs = setup_srs(64)
        coeffs1 = [secrets.randbelow(FR_MODULUS) for _ in range(25)]
        coeffs2 = [secrets.randbelow(FR_MODULUS) for _ in range(17)]
        z1 = secrets.randbelow(FR_MODULUS)
        z2 = secrets.randbelow(FR_MODULUS)

        reg = KernelRegistry()
        register_cpu_kernels(reg)
        exe = Executor(registry=reg)
        g = Graph()
        g.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=srs)
        g.add_buffer(id="coeffs_list", device=Device.CPU, dtype=DType.OBJ, data=[coeffs1, coeffs2])
        g.add_buffer(id="z_list", device=Device.CPU, dtype=DType.OBJ, data=[int(z1), int(z2)])
        g.add_node(op=OpType.KZG_BATCH_OPEN, inputs=["srs", "coeffs_list", "z_list"], outputs=["y_list", "pi_list"])
        exe.run(g, keep=["y_list", "pi_list"])

        y1, pi1 = open_proof(srs, coeffs1, z1)
        y2, pi2 = open_proof(srs, coeffs2, z2)

        ys = g.buffers["y_list"].data
        pis = g.buffers["pi_list"].data
        self.assertEqual(int(ys[0]) % FR_MODULUS, int(y1) % FR_MODULUS)
        self.assertEqual(int(ys[1]) % FR_MODULUS, int(y2) % FR_MODULUS)
        self.assertEqual(pis[0], pi1)
        self.assertEqual(pis[1], pi2)

    def test_kzg_open_prep_batch_plus_commit_matches_reference(self):
        srs = setup_srs(64)
        coeffs1 = [secrets.randbelow(FR_MODULUS) for _ in range(25)]
        coeffs2 = [secrets.randbelow(FR_MODULUS) for _ in range(17)]
        z1 = secrets.randbelow(FR_MODULUS)
        z2 = secrets.randbelow(FR_MODULUS)

        reg = KernelRegistry()
        register_cpu_kernels(reg)
        exe = Executor(registry=reg)
        g = Graph()
        g.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=srs)
        g.add_buffer(id="coeffs_list", device=Device.CPU, dtype=DType.OBJ, data=[coeffs1, coeffs2])
        g.add_buffer(id="z_list", device=Device.CPU, dtype=DType.OBJ, data=[int(z1), int(z2)])
        g.add_node(op=OpType.KZG_OPEN_PREP_BATCH, inputs=["srs", "coeffs_list", "z_list"], outputs=["y_list", "q_list"])
        g.add_node(op=OpType.KZG_BATCH_COMMIT, inputs=["srs", "q_list"], outputs=["pi_list"])
        exe.run(g, keep=["y_list", "pi_list"])

        y1, pi1 = open_proof(srs, coeffs1, z1)
        y2, pi2 = open_proof(srs, coeffs2, z2)

        ys = g.buffers["y_list"].data
        pis = g.buffers["pi_list"].data
        self.assertEqual(int(ys[0]) % FR_MODULUS, int(y1) % FR_MODULUS)
        self.assertEqual(int(ys[1]) % FR_MODULUS, int(y2) % FR_MODULUS)
        self.assertEqual(pis[0], pi1)
        self.assertEqual(pis[1], pi2)


if __name__ == "__main__":
    unittest.main()
