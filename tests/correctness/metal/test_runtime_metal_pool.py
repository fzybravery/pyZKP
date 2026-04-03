import unittest
from runtime.ir.types import DType, Device
from runtime.memory import MemoryPool
from runtime.metal import metal_available, MetalRuntime

@unittest.skipUnless(metal_available(), "Metal runtime not available")
class TestMetalMemoryPool(unittest.TestCase):
    def test_metal_pool_reuse(self):
        pool = MemoryPool()
        rt = MetalRuntime.create_default()
        
        # Alloc first buffer
        buf1 = pool.alloc_metal(rt, 1024)
        self.assertIsNotNone(buf1)
        self.assertEqual(pool.metal_stats.alloc_calls, 1)
        self.assertEqual(pool.metal_stats.reuse_calls, 0)
        self.assertEqual(pool.metal_stats.in_use, 1)
        
        # Alloc second buffer
        buf2 = pool.alloc_metal(rt, 1024)
        self.assertEqual(pool.metal_stats.alloc_calls, 2)
        self.assertEqual(pool.metal_stats.reuse_calls, 0)
        self.assertEqual(pool.metal_stats.in_use, 2)
        
        # Release buf1
        pool.release_metal(buf1)
        self.assertEqual(pool.metal_stats.in_use, 1)
        
        # Alloc again, should reuse buf1
        buf3 = pool.alloc_metal(rt, 1024)
        self.assertEqual(pool.metal_stats.alloc_calls, 3)
        self.assertEqual(pool.metal_stats.reuse_calls, 1)
        self.assertEqual(pool.metal_stats.in_use, 2)
        self.assertEqual(id(buf1), id(buf3))

    def test_cpu_pool_reuse(self):
        pool = MemoryPool()
        
        arr1 = pool.alloc_cpu(DType.FR, 100)
        self.assertEqual(len(arr1), 100)
        self.assertEqual(pool.cpu_stats.alloc_calls, 1)
        self.assertEqual(pool.cpu_stats.in_use, 1)
        
        pool.release_cpu(DType.FR, arr1)
        self.assertEqual(pool.cpu_stats.in_use, 0)
        
        arr2 = pool.alloc_cpu(DType.FR, 100)
        self.assertEqual(pool.cpu_stats.reuse_calls, 1)
        self.assertEqual(id(arr1), id(arr2))

if __name__ == "__main__":
    unittest.main()
