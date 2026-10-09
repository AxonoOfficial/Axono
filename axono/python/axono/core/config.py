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
"""全局 inplace 开关 (默认关闭)。

用法::

    import axono
    axono.set_inplace_enabled(True)   # 打开后 add/matmul/relu 自动原地
    axono.set_inplace_enabled(False)  # 恢复 (默认状态)

开关打开时, 高层算子 (add/matmul/relu) 会检查当前调用是否支持 inplace
(同形、同设备、同 dtype), 支持则直接原地执行, 避免输出分配;
不支持 (如广播、形状不同) 时自动回退到非原地版本, 语义不变。
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_enabled = False  # 默认关闭


def set_inplace_enabled(flag: bool) -> None:
    """设置全局 inplace 开关 (默认 False)。"""
    global _enabled
    with _lock:
        _enabled = bool(flag)


def is_inplace_enabled() -> bool:
    """查询全局 inplace 开关状态。"""
    with _lock:
        return _enabled


def inplace_enabled() -> bool:
    """别名, 同 is_inplace_enabled。"""
    return is_inplace_enabled()
