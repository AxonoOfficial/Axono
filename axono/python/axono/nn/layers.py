# axono/nn/layers.py — v0.2
from __future__ import annotations

import numpy as np

from ..core import DataType, Tensor, get_default_device
from .module import Module


class Linear(Module):
    """全连接层: y = x @ W^T + b (Kaiming 初始化)。"""

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
    """查表 embedding: ids (INT64, 任意形状) -> (..., hidden)。"""

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
        import axono

        return axono.embedding(ids, self._parameters["weight"])


def _default_like(other: Tensor) -> Tensor:
    """建一个与 other 同形状/dtype/device 的未初始化张量 (load_state_dict 用)。"""
    return Tensor.zeros(
        other.shape, dtype=other.dtype, device=other.device if other.device else None
    )


def _dtype_of(arr: np.ndarray) -> DataType:
    return DataType.FLOAT32 if arr.dtype == np.float32 else DataType.FLOAT64
