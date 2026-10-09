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
"""CUDA Graph 支持 —— 消除小算子序列的 CPU 提交开销。

典型收益: 把一段由几十个 kernel 组成的推理前向 (如 6 层 MLP) 捕获成一张
图, 之后每次 replay 只发一次 launch, CPU 端开销从 ~O(算子数) 降到 ~O(1),
小 batch / 小算子的场景下常见 2~10x 端到端加速。

用法::

    import axono
    g = axono.CUDAGraph()

    # 1) 先用真实输入跑一次 (warmup, 同时确定所有中间缓冲区)
    y = model(x)
    g.capture(lambda: model(x))       # 2) 捕获
    g.replay()                        # 3) 回放 (可重复调用)

    # 上下文管理器写法
    with axono.cuda_graph() as g:
        y = model(x)
    g.replay()

约束 (与 CUDA 原生一致):
- 捕获期间不能有 host<->device 拷贝、不能有显式 cudaDeviceSynchronize、
  不能新建/释放显存 —— 因此捕获前需 warmup 一次, 让所有缓冲区先分配好;
- 输入的地址在捕获后固定, 回放前应把新数据写进同一张量 (原地 copy) 而非
  重建张量, 否则图仍读旧地址。

本模块是 C++ 绑定类 ``libaxono.CUDAGraph`` 的 Python 友好封装。
"""

from __future__ import annotations

import libaxono as _l


class CUDAGraph:
    """CUDA Graph 的 Python 包装 (无 CUDA 构建下实例化即报错)。"""

    def __init__(self) -> None:
        if not _l.cuda_available():
            raise RuntimeError("CUDA Graph 需要启用 CUDA 的构建")
        self._impl = _l.CUDAGraph()

    def capture(self, fn) -> "CUDAGraph":
        """捕获 ``fn()`` 内的所有 CUDA 算子调用为一张图。

        内部流程 (C++ 绑定层实现):
        1. warmup —— 先以登记模式跑一遍 ``fn()``, 其间所有 CUDA 分配
           被登记进捕获缓存池;
        2. 正式捕获 —— 重跑 ``fn()``, 同尺寸分配命中缓存 (地址不变),
           图内只含 kernel 节点 (实测部分 CUDA 版本上, 含 alloc 节点的
           图无法重复 launch);
        3. ``fn`` 的返回值保存到 ``self.output``, 回放后从该缓冲区读结果。
        """
        holder: dict = {}

        def _body():
            holder["out"] = fn()

        self._impl.capture(_body)
        self._output = holder.get("out")
        return self

    def replay(self) -> "CUDAGraph":
        """回放已捕获的图 (CPU 端只发一次 launch)。"""
        self._impl.replay()
        return self

    @property
    def output(self):
        """捕获时 ``fn`` 的返回值 (回放后其数据被原地更新)。"""
        return getattr(self, "_output", None)

    def sync(self) -> "CUDAGraph":
        """等待图中所有 kernel 完成。"""
        self._impl.sync()
        return self

    def reset(self) -> None:
        """释放图资源 (可重新 capture)。"""
        self._impl.reset()

    @property
    def is_captured(self) -> bool:
        return self._impl.is_captured

    @property
    def num_nodes(self) -> int:
        """图中 kernel 节点数。"""
        return self._impl.num_nodes


class cuda_graph:
    """上下文管理器: ``with axono.cuda_graph() as g: ...`` 捕获代码块。

    与 ``CUDAGraph.capture(fn)`` 等价, 但允许把捕获体写成普通语句块。
    退出 with 时才真正结束捕获。
    """

    def __init__(self) -> None:
        if not _l.cuda_available():
            raise RuntimeError("CUDA Graph 需要启用 CUDA 的构建")
        self.graph = CUDAGraph()
        self._stream = None

    def __enter__(self) -> CUDAGraph:
        stream = _l._graph_begin_capture()
        self._stream = stream
        return self.graph

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            _l._graph_abort_capture(self._stream)
            return False
        _l._graph_end_capture(self._stream, self.graph._impl)
        return False
