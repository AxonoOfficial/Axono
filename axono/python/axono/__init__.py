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
"""
Axono - High Performance Computing Library
"""

from .core import (
    DataType,
    Status,
    Tensor,
    cuda_available,
    get_default_device,
    operators,
    set_default_device,
)
from .core.config import (
    is_inplace_enabled,
    set_inplace_enabled,
)
from .core.ops import relu, relu_
from .core.operators import add, matmul

__version__ = "0.2.0"
__author__ = "ByteRainLab"
__description__ = "High performance computing library for big data processing"

__all__ = [
    "DataType",
    "Status",
    "Tensor",
    "operators",
    "relu",
    "relu_",
    "add",
    "matmul",
    "cuda_available",
    "get_default_device",
    "set_default_device",
    "set_inplace_enabled",
    "is_inplace_enabled",
    "welcome",
]


def welcome():
    text = (
        "                                                                                                                                                  "
        "\n       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,                                              88"
        '\n      d88b         Y8,    ,8P  d8"\'    `"8b   8888b     88   d8"\'    `"8b                                             ""'
        "\n     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b"
        "\n    d8'  `8b          Y88P    88          88  88  `8b   88  88          88       ,adPPYba,  8b,dPPYba,    ,adPPYb,d8  88  8b,dPPYba,    ,adPPYba,"
        '\n   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88      a8P_____88  88P\'   `"8a  a8"    `Y88  88  88P\'   `"8a  a8P_____88'
        '\n  d8""""""""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P      8PP"""""""  88       88  8b       88  88  88       88  8PP"""""""'
        '\n d8\'        `8b    d8\'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P       "8b,   ,aa  88       88  "8a,   ,d88  88  88       88  "8b,   ,aa'
        '\nd8\'          `8b  8P        Y8  `"Y8888Y"\'    88      `888    `"Y8888Y"\'         `"Ybbd8"\'  88       88   `"YbbdP"Y8  88  88       88   `"Ybbd8"\''
        "\n                                                                                                          aa,    ,88"
        '\n                                                                                                           "Y8bbdP"'
        "\nDear 使用者"
        "\n"
        "\n引擎版本: {__version__}"
        "\n欢迎使用Axono Ai 引擎~"
        "\n"
        "\nAxono的官方团队为您送上诚挚的问候!"
        "\n"
        "\nBest regards,"
        "\n{__author__}"
    )
    text = text.replace("{__version__}", __version__)
    text = text.replace("{__author__}", __author__)
    print(text)
