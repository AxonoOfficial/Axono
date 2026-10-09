# axono/nn/layers.py — v0.2
from __future__ import annotations

import numpy as np

from ..core import Tensor, get_default_device
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
        output = x @ self._parameters["weight"].T
        if self._parameters["bias"] is not None:
            output = output + self._parameters["bias"]
        return output
