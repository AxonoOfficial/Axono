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

> [查看Benchmark](benchmark.md)
## 主要特性 (v0.2)
- 支持的数据精度（见 `axono.core -> DataType`）：
  - int8、int16、int32、int64、**float16**、float32、float64
- 张量抽象（`axono.core -> Tensor`）：
  - 浅拷贝语义、零拷贝借用构造（`Tensor.from_numpy_ref`）、跨设备 CopyFrom 直拷
  - `Tensor.cast_to()` 类型转换（含 float16 ↔ float32，CPU/CUDA）
- `axono.nn` 模块系统：
  - `nn.Linear / nn.Embedding / nn.LayerNorm / nn.RMSNorm` 等
  - `nn.compile(model)` 计算图优化
  - `state_dict / load_state_dict`（支持 memmap 零拷贝加载）
  - `Module.cast_linear_fp16()` 一键 FP16 混合推理（tensor core，权重显存减半）
- 常用算子（`axono.core.ops` / `axono.core.operators`）：
  - matmul（`@` 运算符，cuBLASLt 加速）、add（`+`）
  - 激活函数：`relu / gelu / gelu_tanh / silu / softmax / log_softmax`
  - 融合算子 `axono.fused`：`silu_mul`、`add_rms_norm`（CPU 2.0-2.2x / CUDA 1.6-2.4x）
  - LLM 算子：SDPA（原生 GQA）、RoPE / M-RoPE、KV Cache、`axono.gqa_decode_attention`（split-K flash-decoding，长上下文 1.5-2.0x）
- 权重格式 `axono.format`（AXM1）：
  - `save_axm / load_axm`：AXM1 magic + JSON 索引 + 64B 对齐，numpy memmap 零拷贝映射
  - 权重解析比 safetensors 快 **25x**，CUDA 全模型加载 9.7s → 6.0s
  - 可选 fp16 存储（体积减半），`AxmReader` 流式读取
- CUDA 加速：
  - cuBLASLt（fp32/fp64/fp16 tensor core）+ CUDAGraph 捕获回放
  - caching allocator（2 GiB 预算，FIFO 淘汰）
  - 融合 kernel 库 `axono.fused`
- NumPy 互操作：`Tensor.to_numpy()`、`Tensor.from_numpy(...)`、`Tensor.from_numpy_ref(...)`（零拷贝）
- 设备支持：
  - CPU（OpenBLAS + OpenMP）
  - NVIDIA GPU：`cuda:<id>`

## 安装（Linux）
```bash
# 编译安装
bash build.sh
```
```python
# 已构建安装
# cuda 11.8
pip install axono --index-url=https://download.axono.org/whl/cu118/
# cuda 12.6
pip install axono --index-url=https://download.axono.org/whl/cu126/
# cuda 12.5
pip install axono --index-url=https://download.axono.org/whl/cu125/
# cpu
pip install axono --index-url=https://download.axono.org/whl/cpu/
```
> Windows 系列暂未测试设备，故不提供安装方法。

## 快速上手
```python
import numpy as np
import axono
from axono import nn, Tensor
from axono.core.tensor import DataType

# —— 张量与算子 ——
a = Tensor.from_numpy(np.random.rand(4, 8).astype(np.float32)).to("cuda")
b = a @ a.T                                  # cuBLASLt matmul
h = a.cast_to(DataType.FLOAT16)              # fp16 转换

# —— 构建模型 ——
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
    def forward(self, x):
        return self.fc2(axono.gelu(self.fc1(x)))

model = MLP()
# model.compile()           # 计算图优化 (可选)
# model.cast_linear_fp16()  # FP16 混合推理 (可选, 权重显存减半)
```

## 端到端示例：Qwen3-VL-2B 多模态推理
仓库内置完整的 Qwen3-VL-2B 复刻实现（视觉塔 + 文本塔 + M-RoPE + KV Cache + GQA decode）：

```bash
cd axono/python

# 1. 一键转换 HF 权重为 .axm (可选 fp16 存储减半)
python -m examples.vlm.qwen3vl.convert --model /path/to/qwen3vl-2b [--dtype float16]

# 2. 交互式对话 (自动探测 model.axm)
python -m examples.vlm.qwen3vl.chat --device cuda \
    --model /path/to/qwen3vl-2b \
    --image demo.png --prompt "描述这张图片" \
    [--fp16-linear] [--no-kv-cache] [--no-gqa]

# 3. 端到端前向精度验证 (对照 HF transformers)
python -m examples.vlm.qwen3vl.run_e2e --device cuda
```

## 单元测试
```bash
cd axono/python/tests
python run.py
```
