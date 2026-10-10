# axono/nn/module.py — v0.2
# 轻量 nn 模块基类: 纯 Python 管理参数, 底层 C++ Module 仅做权重容器同步。
from __future__ import annotations

from typing import Dict

from libaxono import Module as _Module

from ..core import Tensor


class Module:
    """轻量 nn 模块基类: 管理参数字典并维护底层 C++ Module 同步。"""

    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._submodules: Dict[str, "Module"] = {}
        self._cpp_module = _Module()
        self._is_training = True
        self._name = self.__class__.__name__

    def add_submodule(self, name: str, module: "Module") -> "Module":
        """注册子模块 (其参数会并入 parameters()/weights())。"""
        if not isinstance(module, Module):
            raise TypeError(f"add_submodule 需要 Module, 得到 {type(module)}")
        self._submodules[name] = module
        return module

    def __setattr__(self, name: str, value):
        """自动注册: Module 属性 -> 子模块; Tensor -> 参数;
        list/tuple of Module -> 带索引的子模块集合 (Torch ModuleList 语义)。"""
        if name.startswith("_") or name in ("training",):
            object.__setattr__(self, name, value)
            return
        submods = self.__dict__.get("_submodules")
        if submods is not None:
            if isinstance(value, Module):
                submods[name] = value
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, Module) for v in value)
            ):
                for i, v in enumerate(value):
                    submods[f"{name}.{i}"] = v
            elif isinstance(value, Tensor):
                params = self.__dict__.get("_parameters")
                if params is not None:
                    params[name] = value
        object.__setattr__(self, name, value)

    def add_weight(self, name: str, tensor: Tensor) -> None:
        """注册一个参数张量 (同时同步到底层 C++ Module)。"""
        if not isinstance(tensor, Tensor):
            raise TypeError(f"add_weight 需要 axono.Tensor, 得到 {type(tensor)}")
        self._parameters[name] = tensor
        self._cpp_module.add_weight(name, tensor)

    def parameters(self) -> Dict[str, Tensor]:
        """返回 {名称: Tensor} 参数字典 (含子模块, 名字用点号连接)。"""
        out: Dict[str, Tensor] = {}
        for name, t in self._parameters.items():
            out[name] = t
        for sub_name, sub in self._submodules.items():
            for pname, pt in sub.parameters().items():
                out[f"{sub_name}.{pname}"] = pt
        return out

    def weights(self):
        """返回所有参数 Tensor 的列表 (含子模块, 按注册顺序)。"""
        weights = [t for t in self._parameters.values() if t is not None]
        for sub in self._submodules.values():
            weights.extend(sub.weights())
        return weights

    @property
    def weight(self) -> Tensor:
        return self._parameters["weight"]

    @property
    def bias(self):
        return self._parameters["bias"]

    def state_dict(self) -> dict:
        """返回 {扁平名: Tensor} 参数字典 (含子模块, 点号连接)。"""
        return dict(self.parameters())

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        """从 {扁平名: Tensor|ndarray} 加载参数 (原地 copy, 保持设备)。"""
        import numpy as np

        params = self.parameters()
        missing = []
        for name, t in params.items():
            if t is None:
                continue  # 无 bias 层 (bias=False)
            if name not in sd:
                missing.append(name)
                continue
            v = sd[name]
            if not isinstance(v, Tensor):
                v = Tensor.from_numpy(np.ascontiguousarray(v.astype(np.float32)))
            src = (
                v.to("cpu").to_numpy()
                if v.device != "cpu" and v.device
                else v.to_numpy()
            )
            if t.device != "cpu" and t.device:
                dst = Tensor.from_numpy(np.ascontiguousarray(src)).to(t.device)
                t.copy_from(dst)
            else:
                t.copy_from_numpy(src)
        if strict and missing:
            raise KeyError(f"state_dict 缺少参数: {missing}")

    def train(self, mode: bool = True) -> "Module":
        self._is_training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def forward(self, *args, **kwargs):  # pragma: no cover - 由子类实现
        raise NotImplementedError("Module 子类需实现 forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        init_args = []
        if hasattr(self, "_init_args"):
            init_args = [f"{k}={v}" for k, v in self._init_args.items()]
        if init_args:
            return f"{cls_name}({', '.join(init_args)})"
        return f"{cls_name}()"
