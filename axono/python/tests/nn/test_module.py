# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""nn.Module / nn.Linear 测试。"""

import numpy as np
import pytest

import axono
from axono.nn import Linear, Module


class TinyModel(Module):
    """两层 MLP, 用于前向传播冒烟。"""

    def __init__(self, in_features, hidden, out_features, device=None):
        super().__init__()
        self.fc1 = Linear(in_features, hidden, device=device)
        self.fc2 = Linear(hidden, out_features, device=device)

    def forward(self, x):
        return self.fc2(axono.relu(self.fc1(x)))


class TestLinear:
    def test_linear_init_shape(self, device):
        layer = Linear(4, 3, device=device)
        w = layer.parameters()["weight"].to_numpy()
        assert w.shape == (3, 4)
        b = layer.parameters()["bias"].to_numpy()
        assert b.shape == (3,)

    def test_linear_forward_shape(self, device, rng):
        layer = Linear(8, 2, device=device)
        x = axono.Tensor.from_numpy(
            rng.standard_normal((5, 8)).astype(np.float32)
        ).to(device)
        y = layer(x)
        assert y.to_numpy().shape == (5, 2)

    def test_linear_forward_values(self, device, rng):
        """手工设置权重后, 前向结果应与 numpy 手算一致: y = x @ W^T + b。"""
        layer = Linear(3, 2, device=device)
        w = rng.standard_normal((2, 3)).astype(np.float32)
        b = rng.standard_normal((2,)).astype(np.float32)
        layer.parameters()["weight"].copy_from_numpy(w)
        layer.parameters()["bias"].copy_from_numpy(b)
        x = rng.standard_normal((4, 3)).astype(np.float32)
        y = layer(axono.Tensor.from_numpy(x).to(device)).to_numpy()
        np.testing.assert_allclose(y, x @ w.T + b, rtol=1e-4, atol=1e-4)

    def test_linear_no_bias(self, device, rng):
        layer = Linear(3, 2, bias=False, device=device)
        assert layer.parameters()["bias"] is None
        x = axono.Tensor.from_numpy(
            rng.standard_normal((2, 3)).astype(np.float32)
        ).to(device)
        assert layer(x).to_numpy().shape == (2, 2)


class TestModule:
    def test_module_forward_mlp(self, device, rng):
        model = TinyModel(4, 8, 2, device=device)
        x = axono.Tensor.from_numpy(
            rng.standard_normal((3, 4)).astype(np.float32)
        ).to(device)
        y = model(x)
        assert y.to_numpy().shape == (3, 2)
        assert np.isfinite(y.to_numpy()).all()

    def test_module_weights_device(self, device):
        model = TinyModel(2, 4, 2, device=device)
        weights = model.weights()
        assert len(weights) > 0
        for w in weights:
            assert w.device == device

    def test_set_default_device(self):
        axono.set_default_device("cpu")
        assert axono.get_default_device() == "cpu"
        t = axono.Tensor(shape=[2, 2])
        assert t.device == "cpu"

    def test_repr(self):
        layer = Linear(4, 3)
        assert "Linear" in repr(layer)
