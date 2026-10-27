```
                                                                                                             
       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,                        ,a8888a,     
      d88b         Y8,    ,8P  d8"'    `"8b   8888b     88   d8"'    `"8b                     ,8P"'  `"Y8,   
     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b                   ,8P        Y8,  
    d8'  `8b          Y88P    88          88  88  `8b   88  88          88      8b       d8  88          88  
   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88      `8b     d8'  88          88  
  d8""""""""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P       `8b   d8'   `8b        d8'  
 d8'        `8b    d8'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P         `8b,d8'     `8ba,  ,ad8'   
d8'          `8b  8P        Y8  `"Y8888Y"'    88      `888    `"Y8888Y"'            "8"         "Y8888P"     
                                                                                                             
```

> PS: 官方Q群 1014082546

Axono 是一个轻量级的人工智能算法库，旨在为教学、研究与原型开发提供简洁可扩展的张量与算子接口。

> [查看 Benchmark 脚本](axono/python/axono/benchmarks/)
## 主要特性 (v0.2)
- 支持的数据精度（见 `axono.core -> DataType`）：
  - int8、int16、int32、int64、**float16**、float32、float64
- 张量抽象（`axono.core -> Tensor`）：
  - 浅拷贝语义、零拷贝借用构造（`Tensor.from_numpy_ref`）、跨设备 CopyFrom 直拷
  - `Tensor.cast_to()` 类型转换（含 float16 ↔ float32，CPU/CUDA）
- `axono.nn` 模块系统：
  - `nn.Linear / nn.Embedding / nn.LayerNorm / nn.RMSNorm` 等
  - `axono.compile(model)` CUDA Graph 捕获与回放（仅 CUDA）
  - `state_dict / load_state_dict`（支持从 NumPy 数组或 memmap 复制到模型参数）
  - `Module.cast_linear_fp16()` 一键 FP16 混合推理（tensor core，权重显存减半）
- 常用算子（`axono.core.ops` / `axono.core.operators`）：
  - matmul（`@` 运算符，cuBLASLt 加速）、add（`+`）
  - 激活函数：`relu / gelu / gelu_tanh / silu / softmax / log_softmax`
  - 融合算子 `axono.fused`：`silu_mul`、`add_rms_norm`
  - LLM 算子：SDPA（原生 GQA）、RoPE / M-RoPE、KV Cache、`axono.gqa_decode_attention`（split-K flash-decoding）
- 权重格式 `axono.format`（AXM1）：
  - `save_axm / load_axm`：AXM1 magic + JSON 索引 + 64B 对齐，numpy memmap 零拷贝映射
  - 可选 fp16 存储（体积减半），`AxmReader` 流式读取
- CUDA 加速：
  - cuBLASLt（FP32/FP64 矩阵乘法及 FP16 tensor core 路径）+ CUDAGraph 捕获回放
  - caching allocator（2 GiB 预算，FIFO 淘汰）
  - 融合 kernel 库 `axono.fused`
- NumPy 互操作：`Tensor.to_numpy()`、`Tensor.from_numpy(...)`、`Tensor.from_numpy_ref(...)`（零拷贝）
- 设备支持：
  - CPU（可选 BLAS/OpenBLAS 与 OpenMP 加速）
  - NVIDIA GPU：`cuda:<id>`

性能取决于硬件、构建选项、输入规模和数据类型，请在目标环境运行基准脚本后比较。

## 安装（Linux）
源码构建需要 Python 3.10+、pip、CMake 和 C++17 编译器；CUDA 构建另需 CUDA Toolkit。以下构建命令在仓库根目录执行。

```bash
# 源码编译安装 (自动检测 CUDA 兼容环境; --cuda 强制启用 / --no-cuda 纯 CPU)
bash install.sh
```
```bash
# 导出 wheel 安装包 (无 GPU 的编译机可加 --cuda --ignore-drivers;
# AXONO_CUDA_TAG=cu126 时 CUDA wheel 文件名带 +cu126 本地版本段)
bash wheel.sh --outdir dist
python -m pip install dist/axono-*.whl
```

也可从 [GitHub Releases](https://github.com/AxonoOfficial/Axono/releases) 下载已构建的 wheel。选择匹配操作系统、Python 版本和 CUDA 版本的文件，再用 `python -m pip install <下载的完整文件路径>` 安装；文件名不带 `+cu...` 的是 CPU 构建。CUDA wheel 运行时仍需要兼容的驱动和 CUDA 运行库。

## 安装（Windows, PowerShell）
```powershell
# 源码编译安装 (自动检测 CUDA; -Cuda 强制 / -NoCuda 纯 CPU)
.\install.ps1
```
```powershell
# 导出 wheel 安装包 (无 GPU 的编译机可加 -Cuda -IgnoreDrivers)
.\wheel.ps1 -OutDir dist
python -m pip install (Get-Item .\dist\axono-*.whl).FullName
```
> 需要 Visual Studio Build Tools (C++ 工作负载); CUDA 构建另需 CUDA
> Toolkit (nvcc), 编译本身不需要 GPU。

## 快速上手
以下示例需要 CUDA 构建、兼容驱动和 GPU。仅使用 CPU 时，将示例中的 `"cuda"` 改为 `"cpu"`，并跳过 CUDA Graph 编译和 FP16 Linear 推理。

```python
import numpy as np
import axono
from axono import nn, Tensor
from axono.core.tensor import DataType

# —— 张量与算子 ——
a = Tensor.from_numpy(np.random.rand(4, 8).astype(np.float32)).to("cuda")
b = a @ a.T  # cuBLASLt matmul
h = a.cast_to(DataType.FLOAT16)  # fp16 转换


# —— 构建模型 ——
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(axono.gelu(self.fc1(x)))


axono.set_backend("cuda")  # 在 CUDA 上创建模型参数
model = MLP()
# model = axono.compile(model)  # CUDA Graph 捕获与回放 (可选，仅 CUDA)
# model.cast_linear_fp16()  # FP16 混合推理 (可选, 权重显存减半)
```

## 端到端示例：Qwen3-VL-2B 多模态推理
仓库提供 Qwen3-VL-2B 图文推理示例（视觉塔 + 文本塔 + M-RoPE + KV Cache + GQA decode）。当前对话接口需要图片，每次问答单独构建上下文，不支持纯文本问答或保留多轮历史。

先安装 Axono，以及示例所需的 `torch`、`torchvision`、`transformers`（需支持 Qwen3-VL）、`safetensors`、`tokenizers` 和 `Pillow`。模型目录需包含配置、tokenizer/processor 文件，以及单文件 `model.safetensors`；当前转换器不读取分片权重索引。以下命令使用绝对模型和图片路径。

```bash
# 从仓库根目录复制示例到独立目录，避免源码包遮蔽已安装的 Axono
AXONO_EXAMPLES=$(mktemp -d)
cp -r axono/python/examples "$AXONO_EXAMPLES/"
cd "$AXONO_EXAMPLES"

# 1. 转换 HF 权重为 .axm；可在命令末尾加 --dtype float16，以 FP16 存储
python -m examples.vlm.qwen3vl.convert --model /path/to/qwen3vl-2b

# 2. 单次图文问答 (自动探测 model.axm)
# 可选参数：--fp16-linear、--no-kv-cache、--no-gqa；省略 --prompt 进入交互模式
python -m examples.vlm.qwen3vl.chat --device cuda \
    --model /path/to/qwen3vl-2b \
    --image /path/to/demo.png --prompt "描述这张图片"

# 3. 端到端前向精度验证 (对照 HF transformers)
python -m examples.vlm.qwen3vl.run_e2e --device cuda --model /path/to/qwen3vl-2b
```

## 单元测试
回到仓库根目录，在已安装 Axono 的 Python 环境中运行。`--import-mode=importlib` 避免测试收集时导入未编译的源码包。

```bash
python -m pip install pytest
python -m pytest axono/python/tests --import-mode=importlib -v
```
