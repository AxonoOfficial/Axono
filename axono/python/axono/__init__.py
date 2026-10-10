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

from . import nn  # noqa: E402  — from axono import nn / nn.Linear 风格
from .core import (
    DataType,
    Status,
    Tensor,
    cuda_available,
    get_default_device,
    operators,
    set_default_device,
)
from .core.backend import (
    get_backend,
    get_matmul_backend,
    set_backend,
    set_matmul_backend,
)
from .core.config import (
    is_inplace_enabled,
    set_inplace_enabled,
)
from .core.cuda_graph import CUDAGraph, cuda_graph
from .core.compile import compile
from .core.ops import (
    abs,
    argmax,
    ceil,
    concat,
    cos,
    embedding,
    exp,
    floor,
    gelu,
    gelu_tanh,
    layer_norm,
    linear,
    linear_nobias,
    mrope_cos_sin,
    log,
    log_softmax,
    neg,
    reciprocal,
    relu,
    relu_,
    rms_norm,
    rope,
    rope_thd,
    rope_with_cos_sin,
    round,
    rsqrt,
    scaled_dot_product_attention,
    sigmoid,
    sign,
    silu,
    sin,
    slice_,
    softmax,
    sqrt,
    square,
    tanh,
)
from .core.operators import add, div, matmul, maximum, minimum, mul, pow, sub

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
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "sigmoid",
    "tanh",
    "sin",
    "cos",
    "rsqrt",
    "square",
    "reciprocal",
    "sign",
    "floor",
    "ceil",
    "round",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "maximum",
    "minimum",
    "matmul",
    "softmax",
    "log_softmax",
    "gelu",
    "gelu_tanh",
    "silu",
    "layer_norm",
    "rms_norm",
    "linear",
    "linear_nobias",
    "mrope_cos_sin",
    "embedding",
    "rope",
    "rope_with_cos_sin",
    "rope_thd",
    "scaled_dot_product_attention",
    "concat",
    "slice_",
    "argmax",
    "cuda_available",
    "get_default_device",
    "set_default_device",
    "set_inplace_enabled",
    "is_inplace_enabled",
    "set_backend",
    "get_backend",
    "set_matmul_backend",
    "get_matmul_backend",
    "CUDAGraph",
    "cuda_graph",
    "compile",
    "nn",
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
