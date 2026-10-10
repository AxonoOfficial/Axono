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
"""``axono.compile`` — 模型编译 (torch.compile 风格)。

mode="normal": CUDA Graph 捕获回放 — 首次调用 warmup + 捕获整个 forward,
之后每次调用只回放一张图 (CPU 提交开销 O(算子数) → O(1))。

用法 (与 torch.compile 一致)::

    model = axono.compile(model, mode="normal")
    y = model(x)          # 第一次: warmup + 捕获 (稍慢)
    y = model(x)          # 之后: 图回放 (快)

语义与约束 (与 torch.compile 的 cudagraph 路线一致):
- 输入张量的 shape/dtype/device 每次调用须一致; 变化时自动重新捕获;
- 输入数据在回放前被原地拷入捕获时的静态输入缓冲;
- 输出是捕获时的静态缓冲, 下一次回放会原地覆盖 — 需要保留请自行 clone;
- 非 Tensor 参数 (bool/int/str) 每次调用必须相同, 否则重新捕获;
- 仅 CUDA 设备可用; CPU 输入自动迁到 CUDA (编译目标设备);
- mode 只支持 "normal" (后续可扩展 reduce-overhead / max-autotune)。
"""

from __future__ import annotations

import libaxono as _l

from .cuda_graph import CUDAGraph


class _TensorArg:
    """编译期静态输入槽: 捕获时的张量 + 每次调用前的原地数据刷新。"""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor

    def copy_in(self, value):
        """把新输入的数据原地写入静态缓冲 (D2D/H2D)。"""
        self.tensor.copy_from(value)


def _sig_of(args, kwargs):
    """调用签名: (shapes+dtypes+devices, 标量参数)。用于判断是否需要重捕。"""
    parts = []
    for a in args:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            parts.append(("T", tuple(a.shape), str(a.dtype), str(a.device)))
        else:
            parts.append(("S", repr(a)))
    for k in sorted(kwargs):
        v = kwargs[k]
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            parts.append((k, "T", tuple(v.shape), str(v.dtype), str(v.device)))
        else:
            parts.append((k, "S", repr(v)))
    return tuple(parts)


class CompiledModel:
    """编译后的模型包装 (mode="normal": CUDA Graph 回放)。"""

    def __init__(self, model, mode: str = "normal"):
        if mode != "normal":
            raise ValueError(
                f"axono.compile: 暂不支持 mode={mode!r} (目前只有 'normal')"
            )
        if not _l.cuda_available():
            raise RuntimeError("axono.compile 需要 CUDA 构建")
        self._model = model
        self._mode = mode
        self._graphs: dict = {}  # sig -> (CUDAGraph, [_TensorArg], outputs)
        self._last_sig = None

    # -- 属性/子模块代理 (model.layer / model.cfg 等直接透传) --
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_model"), name)

    def __setattr__(self, name, value):
        if name.startswith("_") or name in ("__call__",):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_model"), name, value)

    def _build(self, args, kwargs):
        """warmup + 捕获一次 forward, 登记静态输入。"""
        static_args = []
        for a in args:
            if hasattr(a, "shape") and hasattr(a, "dtype"):
                static_args.append(_TensorArg(a))
            else:
                static_args.append(a)
        static_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                static_kwargs[k] = _TensorArg(v)
            else:
                static_kwargs[k] = v

        g = CUDAGraph()
        g.capture(lambda: self._model(
            *[s.tensor if isinstance(s, _TensorArg) else s for s in static_args],
            **{
                k: (s.tensor if isinstance(s, _TensorArg) else s)
                for k, s in static_kwargs.items()
            },
        ))
        return g, static_args, static_kwargs

    def __call__(self, *args, **kwargs):
        sig = _sig_of(args, kwargs)
        entry = self._graphs.get(sig)
        if entry is None:
            # 未编译过的签名: warmup 一次 (登记分配) 再捕获
            self._model(*args, **kwargs)  # warmup (真实输入)
            entry = self._build(args, kwargs)
            self._graphs[sig] = entry
        g, static_args, static_kwargs = entry
        # 刷新静态输入 (原地 copy_from)
        for s, a in zip(static_args, args):
            if isinstance(s, _TensorArg):
                s.copy_in(a)
        for k, s in static_kwargs.items():
            if isinstance(s, _TensorArg):
                s.copy_in(kwargs[k])
        g.replay()
        outs = g.outputs
        return outs[0] if len(outs) == 1 else tuple(outs)

    # 供 NN.Module 风格调用 (model.forward(...))
    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __repr__(self):
        inner = repr(object.__getattribute__(self, "_model"))
        return f"CompiledModel(mode={self._mode!r}, graphs={len(self._graphs)}, model={inner})"


def compile(model, mode: str = "normal"):
    """编译模型 (torch.compile 风格)。

    mode="normal": CUDA Graph 捕获回放。首次调用 warmup+捕获, 之后每次
    调用回放一张图 — CPU 提交开销从 O(算子数) 降到 O(1)。
    """
    return CompiledModel(model, mode=mode)
