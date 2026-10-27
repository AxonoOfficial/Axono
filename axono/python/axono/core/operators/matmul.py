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
"""core.operators.matmul — CPU 使用可选 BLAS 或内置 kernel, CUDA 使用 cuBLAS(Lt)。

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

    CPU 的 FLOAT32/FLOAT64 优先使用构建时找到的 BLAS, 无 BLAS 时回退
    内置 kernel。CUDA 默认优先 cuBLASLt; FLOAT32/FLOAT64 可通过
    set_matmul_backend("cublas") 选择经典 cuBLAS, FLOAT16 当前需要 cuBLASLt。

    参数:
        a: (M, K) 左矩阵
        b: (K, N) 右矩阵
        out: 可选输出张量, 形状须为 (M, N), 结果写入 out (零分配)

    返回:
        Tensor: (M, N) 结果。全局 inplace 开关打开且未指定 out 时,
        原地写 a 并返回 a。

    示例::

        y = axono.matmul(x, w)                 # 或 x @ w
        buf = Tensor.zeros([M, N], device="cuda")
        axono.matmul(x, w, out=buf)            # 复用输出缓冲
    """
    if out is None and is_inplace_enabled():
        _l.matmul_(a, b)
        return a
    if out is not None:
        _l.matmul_into(a, b, out)
        return out
    return _l.matmul(a, b)
