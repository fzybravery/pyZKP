# pyZKP: 基于 Apple Metal 异构加速的零知识证明全栈框架

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-success)

pyZKP 是一个专为学术研究与高性能工程验证设计的**端到端零知识证明（Zero-Knowledge Proof, ZKP）全栈框架**。
本项目旨在解决传统 ZKP 框架在纯 Python 环境下性能极度低下的痛点。通过在底层引入**计算图抽象（DAG）**、**UMA 零拷贝内存池**以及 **Apple Metal GPU 异构加速库**，pyZKP 成功实现了 Groth16 和 PLONK 协议证明生成过程的高达数量级的性能飞跃。

---

## 🌟 核心技术亮点 (Highlights)

### 1. 统一且强大的电路前端 (Frontend API)
*   **类 gnark 的抽象层**：提供类似 Go 语言主流框架 gnark 的 Pythonic 电路构建接口，支持 `Add/Sub/Mul/Div/And/Or/Xor` 等丰富算子。
*   **标准密码学组件库**：内置了专为 ZKP 优化的工业级原语（如 **Poseidon Hash**），开发者可通过简单的 API 调用构建极低约束的复杂隐私电路。
*   **自动化 IR 编译**：将前端电路自动降级为独立于后端的中间表示（`CircuitIR`）和 R1CS 约束系统。

### 2. 深度优化的异构加速内核 (Metal GPU Kernels)
*   **MSM (多标量乘法) 双架构对比**：
    *   **V1**：标准的 Windowed Bucket 累加。
    *   **V2**：引入 **Signed-Digit (w-NAF)** 编码将 Bucket 数量减半，并使用 **CSR (压缩稀疏行)** 架构大幅优化 GPU 并发效率。
*   **NTT (数论变换) 双架构对比**：
    *   **V1**：基于 Bit-reversal 和 Butterfly Kernel 的传统 Cooley-Tukey 算法。
    *   **V2**：基于 **Stockham 算法**的深度优化版，避免了耗时的位反转操作，显著减少全局同步与显存交换。

### 3. 智能 DAG 调度器与零拷贝架构 (Executor & Memory Pool)
*   **跨设备拓扑调度**：DAG 执行器能够自动识别算子依赖，在 CPU 和 GPU 之间进行“智能降级（Fallback）”与流水线调度。
*   **UMA 共享内存池**：利用 Apple Silicon 的统一内存架构（UMA），实现 CPU 与 GPU 之间的数据**零拷贝（Zero-Copy）**，在连续批处理证明（Batch Proving）时彻底消除 PCI-e 传输瓶颈，显存复用率极高。
*   **常数折叠预热**：提供 `fixed-base-policy`，自动检测并缓存固定基点，将运行时的椭圆曲线乘法转化为极速的 GPU 查表操作。

### 4. 工业级可观测性 (Chrome Tracing)
*   内置微秒级性能探针，支持将执行轨迹导出为标准 JSON 格式。
*   可在 `chrome://tracing` 中无缝呈现**双泳道异构调度火焰图**，直观剖析系统瓶颈（MSM/NTT）与内存搬运开销。

---

## 📂 项目架构 (Architecture)

```text
pyZKP/
├── frontend/             # 前端电路构建与编译层
│   ├── api/              # 用户层 API (api.py) 与 Witness 求解 (witness.py)
│   │   └── std/          # 高级标准组件库 (如 poseidon.py)
│   ├── circuit/          # 电路 Schema 与变量定义
│   └── ir/               # 编译器中间表示 (CircuitIR)
├── crypto/               # 纯数学与密码学原语 (CPU 参考实现)
│   ├── ecc/ & pairing/   # 椭圆曲线 (BN254) 与配对运算
│   ├── field/            # 有限域算术 (FR) 与 Montgomery 批量求逆
│   ├── hash/             # Poseidon Hash 纯 Python 实现
│   ├── msm/              # CPU Pippenger 算法
│   └── poly/             # CPU NTT / iNTT 算法
├── protocols/            # 零知识证明协议层
│   ├── groth16/          # Groth16 (Setup, Prove, Verify, R1CS/QAP)
│   └── plonk/            # PLONK (Setup, Prove, Verify, Fiat-Shamir)
├── runtime/              # 核心异构调度与执行引擎
│   ├── ir/               # 计算图节点与缓冲区定义
│   ├── kernels/          # 算子注册表 (KernelRegistry)
│   ├── metal/            # Apple Metal GPU 加速内核源码 (.metal)
│   ├── cache.py          # 电路指纹哈希与 Setup 结果持久化
│   ├── executor.py       # DAG 拓扑执行与跨设备切分调度
│   ├── memory.py         # UMA 零拷贝内存池
│   └── trace.py          # 性能探针与 Chrome Tracing 导出
├── benches/              # 基准测试与压测脚本
└── tests/                # 正确性单元测试 (CPU/Metal 全量覆盖)
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境依赖
*   操作系统：macOS (需要 Apple Silicon 或支持 Metal 的 Intel Mac)
*   Python 3.9+
*   依赖包：`pyobjc-framework-Metal`，`py_ecc`

### 2. 运行全量单元测试
框架包含了 50+ 个深度测试用例，覆盖了底层密码学算子、执行器调度、协议正确性以及前端 API：
```bash
PYTHONPATH=. python3 -m unittest discover tests -v
```

### 3. 运行高性能 Benchmark 并导出 Trace
你可以通过灵活的命令行参数，开启/关闭不同的优化策略（如 GPU 后端、图复用、自动预热等），并生成极具说服力的性能分析图表。

**示例：在 Metal 上对 Groth16 执行 5000 规模的批量压测（开启全量优化）**
```bash
PYTHONPATH=. python3 benches/bench.py \
    --scheme groth16 \
    --backend metal \
    --repeat 5000 \
    --batch 2 \
    --fixed-base-policy auto \
    --reuse-graph
```
执行完毕后，你将在 `traces/` 目录下获得一个 `.json` 文件。
打开基于 Chromium 的浏览器，在地址栏输入 `chrome://tracing`，将生成的 JSON 文件拖入其中，即可欣赏到极其清晰的**异构调度火焰图**。

---

## 📝 典型用法：使用 Poseidon Hash 构建电路

借助本框架的高级抽象层，你可以像写普通 Python 代码一样，快速构建工业级的隐私电路：

```python
from frontend.api.api import API, Circuit
from frontend.circuit.schema import public, secret
from frontend.api.std.poseidon import poseidon_circuit

class VotingCircuit(Circuit):
    def __init__(self):
        self.secret_password = secret()
        self.public_salt = public()
        self.expected_hash = public()

    def define(self, api: API) -> None:
        # 1. 拼接哈希输入
        inputs = [self.secret_password, self.public_salt]
        
        # 2. 调用内置的 Poseidon 组件计算哈希 (仅产生约 780 个约束)
        actual_hash = poseidon_circuit(api, inputs)
        
        # 3. 强制约束
        api.AssertIsEqual(actual_hash, self.expected_hash)
```

---

## ⚠️ 性能评估注意事项
本框架的 `Setup` 阶段目前采用纯 Python 实现。由于 Groth16 的 Trusted Setup 涉及极高复杂度的 $G_2$ 扩域运算和非结构化矩阵遍历，这部分在首次运行时耗时较长。
这并非性能缺陷，而是框架**“重 Prove，轻 Setup”**的设计折衷：
1. `Setup` 为一次性离线操作，本框架已实现**持久化缓存 (`save_setup_cache`)**，后续请求耗时为 0。
2. 我们将全部的 Metal GPU 算力与异构调度机制倾斜到了高频且实时性要求极高的 `Prove`（证明生成）阶段。

---

## 📜 许可证 (License)
本项目采用 [MIT License](LICENSE) 开源协议。
