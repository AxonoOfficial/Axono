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
"""core.operators.matmul — CPU 走 AVX kernel, CUDA 走 cuBLAS"""

from libaxono import matmul as _matmul

from ..tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return Tensor.from_raw(_matmul(a._tensor, b._tensor))
