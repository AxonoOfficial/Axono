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
"""nn.KVCache / nn.DynamicKVCache 测试。"""

import numpy as np
import pytest

import axono
from axono import nn


def _kv(seq, n_kv, d, device, seed=0):
    rng = np.random.default_rng(seed)
    t = axono.Tensor.from_numpy(
        rng.standard_normal((seq, n_kv, d)).astype(np.float32)
    )
    return t.to(device) if device != "cpu" else t


class TestDynamicKVCache:
    def test_len_and_get(self, device):
        cache = nn.DynamicKVCache(2, device=device)
        assert len(cache) == 0
        k1, v1 = _kv(3, 2, 4, device, 0), _kv(3, 2, 4, device, 1)
        cache.update(0, k1, v1)
        assert len(cache) == 3
        k, v = cache.get(0)
        assert k.to_numpy().shape == (3, 2, 4)
        np.testing.assert_allclose(k.to_numpy(), k1.to_numpy(), rtol=1e-6)

    def test_append_accumulates(self, device):
        cache = nn.DynamicKVCache(1, device=device)
        cache.update(0, _kv(3, 2, 4, device, 0), _kv(3, 2, 4, device, 1))
        k2 = _kv(1, 2, 4, device, 2)
        v2 = _kv(1, 2, 4, device, 3)
        cache.update(0, k2, v2)
        assert len(cache) == 4
        k, v = cache.get(0)
        kn = k.to_numpy()
        np.testing.assert_allclose(kn[:3], _kv(3, 2, 4, device, 0).to_numpy(),
                                   rtol=1e-6)
        np.testing.assert_allclose(kn[3], k2.to_numpy()[0], rtol=1e-6)
        assert v.to_numpy().shape == (4, 2, 4)

    def test_reset(self, device):
        cache = nn.DynamicKVCache(2, device=device)
        cache.update(0, _kv(3, 2, 4, device, 0), _kv(3, 2, 4, device, 1))
        cache.reset()
        assert len(cache) == 0
        with pytest.raises(RuntimeError):
            cache.get(0)

    def test_get_before_update_raises(self, device):
        cache = nn.DynamicKVCache(1, device=device)
        with pytest.raises(RuntimeError):
            cache.get(0)

    def test_layers_independent(self, device):
        cache = nn.DynamicKVCache(2, device=device)
        cache.update(0, _kv(2, 1, 4, device, 0), _kv(2, 1, 4, device, 1))
        cache.update(1, _kv(5, 1, 4, device, 2), _kv(5, 1, 4, device, 3))
        # 两层独立: 各自序列长度不同, __len__ 取第一个非空层
        assert len(cache) == 2
        assert cache.get(0)[0].to_numpy().shape == (2, 1, 4)
        assert cache.get(1)[0].to_numpy().shape == (5, 1, 4)

    def test_custom_cache_component(self, device):
        """自定义 KVCache 组件: 只保留最近 2 步的滑动窗口。"""

        class SlidingWindowCache(nn.KVCache):
            def __init__(self, window=2):
                self.window = window
                self.k = None
                self.v = None

            def update(self, layer_idx, k_new, v_new):
                if self.k is None:
                    self.k, self.v = k_new, v_new
                else:
                    self.k = axono.concat(self.k, k_new, 0)
                    self.v = axono.concat(self.v, v_new, 0)
                if int(self.k.shape[0]) > self.window:
                    n = int(self.k.shape[0]) - self.window
                    self.k = axono.slice_(self.k, 0, n, self.window)
                    self.v = axono.slice_(self.v, 0, n, self.window)

            def get(self, layer_idx):
                return self.k, self.v

            def reset(self):
                self.k = self.v = None

            def __len__(self):
                return 0 if self.k is None else int(self.k.shape[0])

        cache = SlidingWindowCache(window=2)
        for i in range(4):
            cache.update(0, _kv(1, 1, 4, device, i), _kv(1, 1, 4, device, i + 10))
        assert len(cache) == 2
        np.testing.assert_allclose(
            cache.get(0)[0].to_numpy(),
            np.concatenate(
                [_kv(1, 1, 4, "cpu", 2).to_numpy(),
                 _kv(1, 1, 4, "cpu", 3).to_numpy()]
            ),
            rtol=1e-6,
        )


class TestKVCacheWithAttention:
    """KV cache 与 SDPA 集成: cache 路径结果 == 一次性全序列前向。"""

    def test_decode_matches_full_forward(self, device):
        """q_len=1 + cache 追加 == 全序列 causal 前向的最后一个位置。"""
        rng = np.random.default_rng(42)
        seq, hq, hkv, d = 6, 4, 2, 8
        q_all = rng.standard_normal((seq, hq, d)).astype(np.float32)
        k_all = rng.standard_normal((seq, hkv, d)).astype(np.float32)
        v_all = rng.standard_normal((seq, hkv, d)).astype(np.float32)

        def ton(t):
            return t.to(device) if device != "cpu" else t

        # 参考: 全序列 causal
        ref = axono.scaled_dot_product_attention(
            ton(axono.Tensor.from_numpy(q_all)),
            ton(axono.Tensor.from_numpy(k_all)),
            ton(axono.Tensor.from_numpy(v_all)),
            True,
        ).to_numpy()

        # cache 路径: 逐步 prefill/decode (q=1, k/v 追加, 非因果)
        cache = nn.DynamicKVCache(1, device=device)
        for t in range(seq):
            q = ton(axono.Tensor.from_numpy(q_all[t : t + 1]))
            k = ton(axono.Tensor.from_numpy(k_all[t : t + 1]))
            v = ton(axono.Tensor.from_numpy(v_all[t : t + 1]))
            cache.update(0, k, v)
            kk, vv = cache.get(0)
            out = axono.scaled_dot_product_attention(q, kk, vv, False).to_numpy()
            np.testing.assert_allclose(out[0], ref[t], rtol=1e-4, atol=1e-5)
