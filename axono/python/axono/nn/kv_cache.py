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
"""KV Cache 组件。

注意力层在自回归解码时无需重算全部历史 k/v —— 把每层的 k/v 缓存起来,
每步只对新 token 计算 q/k/v 并追加。本模块提供:

- ``KVCache``: 抽象基类 (自定义组件只需实现 update/reset/__len__);
- ``DynamicKVCache``: 逐层缓存 + concat 追加的默认实现。

用法 (模型侧, 见 examples/vlm/qwen3vl/model.py)::

    cache = nn.DynamicKVCache(n_layers, device="cuda")
    logits = model.forward(..., kv_cache=cache)     # prefill: 写入 prompt 的 k/v
    logits = model.forward(..., kv_cache=cache)     # decode: 只算新 token
    cache.reset()                                    # 开启新一轮对话

自定义组件::

    class SlidingWindowCache(nn.KVCache):
        def update(self, layer_idx, k_new, v_new): ...
        def get(self, layer_idx): ...      # -> (k_full, v_full)
        def reset(self): ...
        def __len__(self): ...
"""

from __future__ import annotations

import axono


class KVCache:
    """KV Cache 抽象基类。

    子类必须实现:
        update(layer_idx, k_new, v_new) -> None   追加新 token 的 k/v;
        get(layer_idx) -> (k, v)                  返回该层完整 k/v (含新 token);
        reset() -> None                           清空 (新一轮对话);
        __len__() -> int                          当前缓存序列长度。
    """

    def update(self, layer_idx: int, k_new, v_new):  # pragma: no cover
        raise NotImplementedError

    def get(self, layer_idx: int):  # pragma: no cover
        raise NotImplementedError

    def reset(self):  # pragma: no cover
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError


class DynamicKVCache(KVCache):
    """默认实现: 每层存 (k, v) Tensor, 新 token 到达时 concat 追加。

    - k/v 形状约定 (seq, n_kv_head, d_head), 与模型 attention 一致;
    - 内部保存「已拼接结果」, 每步只做一次 concat (O(S) 拷贝, 无重复计算);
    - concat 是纯拷贝, 不改变数值 —— 与无 cache 逐位一致。
    """

    def __init__(self, n_layers: int, device: str | None = None):
        self.n_layers = n_layers
        self.device = device
        self._k: list = [None] * n_layers
        self._v: list = [None] * n_layers

    def update(self, layer_idx: int, k_new, v_new) -> None:
        if self._k[layer_idx] is None:
            self._k[layer_idx] = k_new
            self._v[layer_idx] = v_new
            return
        self._k[layer_idx] = axono.concat(self._k[layer_idx], k_new, 0)
        self._v[layer_idx] = axono.concat(self._v[layer_idx], v_new, 0)

    def get(self, layer_idx: int):
        if self._k[layer_idx] is None:
            raise RuntimeError("DynamicKVCache: 尚未 update (prefill 未执行?)")
        return self._k[layer_idx], self._v[layer_idx]

    def reset(self) -> None:
        self._k = [None] * self.n_layers
        self._v = [None] * self.n_layers

    def __len__(self) -> int:
        for k in self._k:
            if k is not None:
                return int(k.shape[0])
        return 0
