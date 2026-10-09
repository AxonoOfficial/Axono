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
"""core.operators.add — 逐元素加法 (支持 (M,K)+(K,) 一维广播)。"""

import numpy as np

from libaxono import add as _add

from ..tensor import Tensor


def _tile_rows(b: Tensor, m: int) -> Tensor:
    """把一维 (K,) 张量在行方向复制 m 次得到 (M, K)。"""
    arr = b.to_numpy()
    tiled = np.tile(arr, (m, 1))
    return Tensor.from_numpy(tiled).to(b.device)


def add(a: Tensor, b: Tensor) -> Tensor:
    """逐元素加法。

    支持两种情形:
    - a 与 b 同形: 逐元素相加 (CPU/CUDA 后端原生);
    - a 为二维 (M, K) 且 b 为一维 (K,): 沿行广播 (Linear bias 场景)。
      CPU 后端原生支持; CUDA 后端通过行复制后相加实现。
    """
    if a.shape == b.shape:
        return Tensor.from_raw(_add(a._tensor, b._tensor))

    # 一维广播: a (M, K) + b (K,)
    if len(a.shape) == 2 and len(b.shape) == 1 and a.shape[1] == b.shape[0]:
        if a.device == "cpu":
            return Tensor.from_raw(_add(a._tensor, b._tensor))
        return Tensor.from_raw(_add(a._tensor, _tile_rows(b, a.shape[0])._tensor))

    raise ValueError(
        f"add: 形状不兼容 {a.shape} vs {b.shape} (仅支持同形或 (M,K)+(K,) 广播)"
    )
