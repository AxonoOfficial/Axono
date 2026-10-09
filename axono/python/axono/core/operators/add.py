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
"""core.operators.add — 逐元素加法 (支持 (M,K)+(K,) 一维广播)。

重构要点:
- 全部走 libaxono 绑定, 无 from_raw 中间包装;
- 支持 out= 指定输出 (真 into, 零额外分配);
- 全局 inplace 开关打开且支持时自动原地 (add_);
- 非 inplace 路径保证输入不被修改 (纯函数语义)。
"""

import numpy as np

import libaxono as _l

from ..config import is_inplace_enabled
from ..tensor import Tensor


def _tile_rows(b: Tensor, m: int) -> Tensor:
    """把一维 (K,) 张量在行方向复制 m 次得到 (M, K)。"""
    arr = b.to_numpy()
    tiled = np.tile(arr, (m, 1))
    return Tensor.from_numpy(tiled).to(b.device)


def _inplace_ok(a: Tensor, b: Tensor) -> bool:
    """检查 a += b 是否可安全原地执行 (同形/同设备/同类型)。"""
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.device == b.device
    )


def add(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
    """逐元素加法, result = a + b。

    支持两种情形:
    - a 与 b 同形: 逐元素相加 (CPU/CUDA 后端原生);
    - a 为二维 (M, K) 且 b 为一维 (K,): 沿行广播 (Linear bias 场景)。

    参数:
        out: 可选输出张量 (形状须与结果一致)。给定后零分配写入 out。

    全局 inplace 开关 (axono.set_inplace_enabled(True)) 打开时,
    若未指定 out 且 a/b 同形同设备同类型, 直接原地 a += b 并返回 a。
    """
    same_shape = a.shape == b.shape
    broadcast = (
        len(a.shape) == 2 and len(b.shape) == 1 and a.shape[1] == b.shape[0]
    )
    if not same_shape and not broadcast:
        raise ValueError(
            f"add: 形状不兼容 {a.shape} vs {b.shape} (仅支持同形或 (M,K)+(K,) 广播)"
        )

    # 广播情形 CUDA 需先 tile, 不支持原地
    need_tile = broadcast and a.device != "cpu"
    can_inplace = (
        out is None and same_shape and not need_tile and is_inplace_enabled()
        and _inplace_ok(a, b)
    )
    if can_inplace:
        _l.add_(a, b)
        return a

    if out is not None and same_shape:
        _l.add_into(a, b, out)
        return out

    # 非原地: 输入不变, libaxono 直接返回新的绑定 Tensor 对象
    if need_tile:
        b_eff = _tile_rows(b, a.shape[0])
    else:
        b_eff = b
    return _l.add(a, b_eff)
