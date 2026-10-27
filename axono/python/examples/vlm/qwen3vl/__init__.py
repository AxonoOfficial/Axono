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
"""Qwen3-VL-2B on Axono: nn.Module 模型 + 单轮图文问答 + benchmark。

先按 README 将 examples 复制到独立目录，在该目录运行:
    python -m examples.vlm.qwen3vl.run_e2e --device cuda --model /path/to/qwen3vl-2b
    python -m examples.vlm.qwen3vl.chat --device cuda --model /path/to/qwen3vl-2b --image /path/to/img.jpg
    python -m examples.vlm.qwen3vl.benchmark_torch --device cuda --model /path/to/qwen3vl-2b
"""

from .chat import Qwen3VLChat  # noqa: F401
from .model import (  # noqa: F401
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
    load_hf_state_dict,
)

__all__ = [
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLTextModel",
    "Qwen3VLVisionModel",
    "Qwen3VLChat",
    "load_hf_state_dict",
]
