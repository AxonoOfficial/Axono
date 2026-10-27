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

把一段 CUDA 算子捕获成一张图, 之后每次 replay 提交一次图并等待执行完成,
减少逐个提交算子的开销。实际收益取决于模型和设备。

用法::

    import axono
    g = axono.CUDAGraph()

    # model 参数和 x 已在同一 CUDA 设备上
    g.capture(lambda: model(x))       # 内部先预热, 再捕获
    g.replay()                        # 回放 (可重复调用)

    # 多输出: fn 返回 tuple, 回放后 g.outputs 中每个缓冲同时更新
    g.reset()                        # 重新捕获前先释放旧图
    g.capture(lambda: (axono.relu(a @ b), axono.tanh(c @ d)))
    g.replay()
    y1, y2 = g.outputs

    # 上下文管理器写法 (块内 g.output = y 指定输出, 也接受 tuple)
    with axono.cuda_graph() as g2:
        y1 = axono.relu(a @ b)
        y2 = axono.tanh(c @ d)
        g2.output = (y1, y2)         # 多输出
    g2.replay()                       # 3) 回放 (可重复调用)

约束:
- 输入准备、host<->device 拷贝和显式设备同步须放在捕获体外;
- capture(fn) 内部先预热并登记 Axono 的张量分配, 再正式捕获;
  with 模式直接捕获代码块, 使用 stream-ordered 分配, 不重跑 Python 代码块;
- 输入的地址在捕获后固定, 回放前应把新数据写进同一张量 (原地 copy) 而非
  重建张量, 否则图仍读旧地址。

本模块是 C++ 绑定类 ``libaxono.CUDAGraph`` 的 Python 友好封装。
"""

from __future__ import annotations

import libaxono as _l


class CUDAGraph:
    """CUDA Graph 的 Python 包装 (无 CUDA 构建下实例化即报错)。

    完整生命周期: capture (内部预热) → 反复 replay → reset。
    模型参数和输入须提前放在同一 CUDA 设备上。
    """

    def __init__(self) -> None:
        if not _l.cuda_available():
            raise RuntimeError("CUDA Graph 需要启用 CUDA 的构建")
        self._impl = _l.CUDAGraph()

    def capture(self, fn) -> CUDAGraph:
        """捕获 ``fn()`` 内的所有 CUDA 算子调用为一张图。

        内部流程 (C++ 绑定层实现):
        1. warmup —— 先以登记模式跑一遍 ``fn()``, 其间所有 CUDA 分配
           被登记进捕获缓存池;
        2. 正式捕获 —— 重跑 ``fn()``, 同尺寸分配命中缓存 (地址不变),
           图内只含 kernel 节点 (实测部分 CUDA 版本上, 含 alloc 节点的
           图无法重复 launch);
        3. ``fn`` 的返回值保存到 ``self.output``, 回放后从该缓冲区读结果。

        多输出: 若 ``fn`` 返回 tuple/list, 则各元素分别保存到
        ``self.outputs`` (list) 且 ``self.output`` 指向第一个; 回放时
        所有输出缓冲原地更新。
        """
        holder: dict = {}

        def _body():
            holder["out"] = fn()

        self._impl.capture(_body)
        self._set_outputs(holder.get("out"))
        return self

    def _set_outputs(self, value) -> None:
        """规范化输出: 单值 -> [v]; tuple/list -> 原样列表。"""
        if isinstance(value, (tuple, list)):
            self._outputs = list(value)
        else:
            self._outputs = [value]
        self._output = self._outputs[0] if self._outputs else None

    def replay(self) -> CUDAGraph:
        """回放已捕获的图, 并等待该图的执行流完成。

        回放前把新输入数据写入静态输入缓冲 (copy_from_numpy / D2D);
        回放后 outputs 指向的缓冲已是新结果。可无限次调用。

        示例::

            g.replay()                       # 每步推理只这一句
            y = g.outputs[0]
        """
        self._impl.replay()
        return self

    @property
    def output(self):
        """捕获时 ``fn`` 的第一个返回值 (回放后其数据被原地更新)。

        注意: 下一次回放会覆盖此缓冲; 可用 g.output.to_numpy().copy()
        保存独立的 CPU 数组。
        """
        return getattr(self, "_output", None)

    @output.setter
    def output(self, value):
        """``with`` 用法: 在捕获块内 ``g.output = y`` 显式指定输出。

        也接受 tuple/list —— 此时等价于 ``g.outputs = (y1, y2)``,
        回放后每个缓冲都被原地更新 (多输出)。
        """
        self._set_outputs(value)

    @property
    def outputs(self):
        """所有输出张量列表 (单输出时为长度 1 的列表)。"""
        return list(getattr(self, "_outputs", []))

    @outputs.setter
    def outputs(self, values):
        self._set_outputs(values)

    def sync(self) -> CUDAGraph:
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
        """图中的总节点数, 包含 kernel 和可能存在的内存操作节点。"""
        return self._impl.num_nodes


class cuda_graph:  # noqa: N801 — 上下文管理器惯用小写命名 (与 threading.Lock 同风格)
    """上下文管理器: ``with axono.cuda_graph() as g: ...`` 捕获代码块。

    进入时开始捕获, 退出时结束捕获并完成图的实例化。与 capture(fn)
    不同, 此方式不预热或重跑 Python 代码块; 临时张量使用 stream-ordered
    分配。输入和模型初始化须在进入代码块前完成。

    用法::

        with axono.cuda_graph() as g:
            y = model(x)
            g.output = y          # 单输出; 多输出: g.output = (y1, y2)
        g.replay()
        print(g.output.to_numpy())
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
