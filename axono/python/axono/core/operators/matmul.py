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
"""core.operators.matmul — CPU 走 AVX kernel, CUDA 走 cuBLAS。

重构要点:
- 无 from_raw 包装;
- 支持 out= 指定输出 (真 into);
- 全局 inplace 开关打开时自动 a = a @ b (matmul_)。
"""

import libaxono as _l

from ..config import is_inplace_enabled
from ..tensor import Tensor


def matmul(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
    """矩阵乘法 result = a @ b (仅二维)。

    参数:
        out: 可选输出张量, 形状须为 (a.shape[0], b.shape[1])。

    全局 inplace 开关打开且未指定 out 时, 直接 a = a @ b 并返回 a。
    """
    if out is None and is_inplace_enabled():
        _l.matmul_(a, b)
        return a
    if out is not None:
        _l.matmul_into(a, b, out)
        return out
    return _l.matmul(a, b)
