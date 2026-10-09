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
"""axono.core — 张量、数据类型与设备管理。"""

import os
import sys

# axono 包目录 (core 的上一级); 扩展可能落在包目录或包目录下的 library/
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _candidate in (_pkg_dir, os.path.join(_pkg_dir, "library")):
    if _candidate not in sys.path:
        sys.path.append(_candidate)

from libaxono import DataType, Status  # noqa: E402
from libaxono import cuda_available  # noqa: E402

from . import operators  # noqa: E402
from .tensor import Tensor, get_default_device, set_default_device  # noqa: E402

__all__ = [
    "DataType",
    "Status",
    "Tensor",
    "cuda_available",
    "operators",
    "get_default_device",
    "set_default_device",
]
