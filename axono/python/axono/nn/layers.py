# axono/nn/layers.py — v0.2
"""神经网络层组件 (nn.Linear / nn.Embedding)。

- ``Linear``: 全连接层 y = x @ W^T + b, Kaiming 初始化, bias 可选。
- ``Embedding``: 词嵌入查表层。

所有层继承 ``nn.Module``, 参数自动注册, 支持 ``load_state_dict`` /
``state_dict`` 与 C++ 侧权重容器同步。
"""
from __future__ import annotations

import numpy as np

from ..core import DataType, Tensor, get_default_device
from .module import Module


class Linear(Module):
    """全连接层: ``y = x @ W^T + b`` (Kaiming 初始化)。

    FP16 模式: cast_linear_fp16() 后 weight 为 FLOAT16, matmul 走
    cuBLASLt tensor core; 输入自动随 weight dtype 转换。

    参数:
        in_features: 输入维度 K
        out_features: 输出维度 N
        bias: 是否含偏置 (False 时 weight-only, 节省一次广播加法)
        device: 参数所在设备, 缺省用全局默认设备

    示例::

        fc = nn.Linear(2048, 1024)          # W: (1024, 2048)
        y = fc(x)                           # x: (B, 2048) → (B, 1024)
        fc.cast_linear_fp16()               # 权重转 FP16 (省一半显存)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        device = device or get_default_device()
        self._init_args = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "device": device,
        }

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        scale = np.sqrt(2.0 / in_features)
        weight_data = np.random.normal(
            loc=0.0, scale=scale, size=(out_features, in_features)
        ).astype(np.float32)
        weight_tensor = Tensor.from_numpy(weight_data).to(device)
        self.add_weight("weight", weight_tensor)

        if bias:
            bias_data = np.zeros(out_features, dtype=np.float32)
            bias_tensor = Tensor.from_numpy(bias_data).to(device)
            self.add_weight("bias", bias_tensor)
        else:
            self._parameters["bias"] = None

    def forward(self, x: Tensor) -> Tensor:
        """前向传播: y = x @ weight.T + bias (若启用)。"""
        import axono

        w = self._parameters["weight"]
        b = self._parameters["bias"]
        if b is not None:
            return axono.linear(x, w, b)
        return axono.linear_nobias(x, w)


class LayerNorm(Module):
    """LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias。"""

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, device: str | None = None
    ):
        super().__init__()
        device = device or get_default_device()
        self._init_args = {"normalized_shape": normalized_shape, "eps": eps}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.add_weight("weight", Tensor.ones((normalized_shape,), device=device))
        self.add_weight("bias", Tensor.zeros((normalized_shape,), device=device))

    def forward(self, x: Tensor) -> Tensor:
        import axono

        return axono.layer_norm(
            x, self._parameters["weight"], self._parameters["bias"], self.eps
        )


class RMSNorm(Module):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight。"""

    def __init__(
        self, normalized_shape: int, eps: float = 1e-6, device: str | None = None
    ):
        super().__init__()
        device = device or get_default_device()
        self._init_args = {"normalized_shape": normalized_shape, "eps": eps}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.add_weight("weight", Tensor.ones((normalized_shape,), device=device))

    def forward(self, x: Tensor) -> Tensor:
        import axono

        return axono.rms_norm(x, self._parameters["weight"], self.eps)


class Embedding(Module):
    """查表 embedding: 输入 token id, 输出词向量。

    参数:
        num_embeddings: 词表大小 V
        embedding_dim: 词向量维度 H
        device: 参数所在设备, 缺省用全局默认设备

    示例::

        emb = nn.Embedding(151936, 2048)          # Qwen3 词表
        ids = Tensor.from_numpy(np.array([[1, 5, 9]], dtype=np.int64))
        h = emb(ids)                              # (1, 3, 2048)
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, device: str | None = None
    ):
        super().__init__()
        device = device or get_default_device()
        self._init_args = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
        }
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = Tensor.randn((num_embeddings, embedding_dim), device=device)
        self.add_weight("weight", w)

    def forward(self, ids: Tensor) -> Tensor:
        """ids (INT64, 任意形状) → (..., embedding_dim) 词向量。"""
        import axono

        return axono.embedding(ids, self._parameters["weight"])


def _default_like(other: Tensor) -> Tensor:
    """建一个与 other 同形状/dtype/device 的未初始化张量 (load_state_dict 用)。"""
    return Tensor.zeros(
        other.shape, dtype=other.dtype, device=other.device if other.device else None
    )


def _dtype_of(arr: np.ndarray) -> DataType:
    return DataType.FLOAT32 if arr.dtype == np.float32 else DataType.FLOAT64
