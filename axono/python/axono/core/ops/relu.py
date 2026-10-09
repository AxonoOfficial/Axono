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
"""core.ops.relu。

重构要点:
- 非原地 relu 保证输入不变, 返回全新张量;
- relu_(x) 真原地: 直接在输入存储上执行 kernel, 无拷贝无临时;
- inplace=True 保持兼容, 等价 relu_(x) 后返回 x;
- 全局 inplace 开关打开时 relu 默认走真原地。
"""

import libaxono as _l

from ..config import is_inplace_enabled
from ..tensor import Tensor


def relu(x: Tensor, inplace: bool | None = None) -> Tensor:
    """ReLU 激活。

    参数:
        inplace: None (默认) 时跟随全局 inplace 开关;
                 True 强制真原地 (修改 x 并返回 x);
                 False 强制非原地 (x 不变, 返回新张量)。
    """
    if inplace is None:
        inplace = is_inplace_enabled()
    if inplace:
        _l.relu_(x)
        return x
    return _l.relu(x)


def relu_(x: Tensor) -> Tensor:
    """真原地 ReLU: 直接修改并返回 x (零分配)。"""
    _l.relu_(x)
    return x
