import secrets
import unittest

from pyZKP.common.crypto.field.fr import FR_MODULUS
from pyZKP.common.crypto.kzg.cpu_ref import commit, open_proof, setup_srs
from pyZKP.runtime import Executor, KernelRegistry, RuntimeConfig
from pyZKP.runtime.ir import Backend, Device, DType, Graph, OpType
from pyZKP.runtime.kernels.cpu import register_cpu_kernels
from pyZKP.runtime.kernels.metal import register_metal_kernels
from pyZKP.runtime.metal import metal_available


@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestRuntimeMetalKZG(unittest.TestCase):
    def test_kzg_batch_commit_matches_single(self):
        srs = setup_srs(64)
        polys = []
        for _ in range(5):
            deg = secrets.randbelow(20) + 1
            polys.append([secrets.randbelow(FR_MODULUS) for _ in range(deg)])

        reg = KernelRegistry()
        register_cpu_kernels(reg, backend=Backend.METAL)
        register_metal_kernels(reg)
        exe = Executor(registry=reg)
        
        g = Graph()
        g.add_buffer(id="srs", device=Device.CPU, dtype=DType.OBJ, data=srs)
        g.add_buffer(id="polys", device=Device.CPU, dtype=DType.OBJ, data=polys)
        g.add_node(op=OpType.KZG_BATCH_COMMIT, inputs=["srs", "polys"], outputs=["cms"])
        
        exe.run(g, runtime_config=RuntimeConfig(backend=Backend.METAL), keep=["cms"])
        cms = g.buffers["cms"].data

        expected = [commit(srs, p) for p in polys]
        from py_ecc.optimized_bn128 import normalize
        for i in range(len(cms)):
            if normalize(cms[i]) != normalize(expected[i]):
                print(f"Mismatch at index {i}: length {len(polys[i])}")
                print(f"Metal: {normalize(cms[i])}")
                print(f"CPU: {normalize(expected[i])}")
            self.assertEqual(normalize(cms[i]), normalize(expected[i]))

if __name__ == "__main__":
    unittest.main()
