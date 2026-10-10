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
"""Axono 原生权重格式 (.axm)。

单文件容器: 自定义头 + 张量索引 + 连续原始字节段。设计目标:
- 加载走 numpy memmap 零拷贝映射 (不做 dtype/格式转换链),
  加载耗时 ≈ 文件读取 + H2D 拷贝本身;
- 支持任意 numpy dtype (默认 fp32, 转换时可选 fp16 减半体积);
- 索引为 JSON, 可流式读取单个张量。

布局::

    offset 0   : magic "AXM1" (4B)
    offset 4   : version uint16 (little-endian, 当前 1)
    offset 6   : 保留 (2B)
    offset 8   : 索引长度 uint64
    offset 16  : 索引 JSON (utf-8)
    offset 16+L: 张量数据 (连续, 64 字节对齐)

索引 JSON: {"tensors": [{"name", "dtype", "shape", "offset", "size"}],
            "meta": {...}}
offset/size 为相对数据段起点的字节。
"""

import json
import struct

import numpy as np

MAGIC = b"AXM1"
VERSION = 1
_HEADER = struct.Struct("<4sHHQ")  # magic, version, reserved, index_len
_ALIGN = 64

__all__ = ["save_axm", "load_axm", "read_axm_index", "AxmReader"]


def _align_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def save_axm(state_dict: dict, path: str, meta: dict | None = None,
             dtype: str | np.dtype | None = None) -> None:
    """把 {name: ndarray} 写入 .axm 文件。

    dtype: 可选统一转换 ("float32"/"float16"/np.dtype); None = 保留原 dtype。
    """
    entries = []
    data_off = 0
    chunks = []
    for name, arr in state_dict.items():
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype is not None \
            else np.ascontiguousarray(arr)
        raw = a.tobytes()
        chunks.append(raw)
        entries.append({
            "name": name,
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "offset": data_off,
            "size": len(raw),
        })
        data_off = _align_up(data_off + len(raw))
    index = {"tensors": entries, "meta": meta or {}}
    idx_bytes = json.dumps(index).encode("utf-8")
    with open(path, "wb") as f:
        f.write(_HEADER.pack(MAGIC, VERSION, 0, len(idx_bytes)))
        f.write(idx_bytes)
        pad = _HEADER.size + len(idx_bytes)
        f.write(b"\0" * (_align_up(pad) - pad))
        for c in chunks:
            f.write(c)
            if len(c) % _ALIGN:
                f.write(b"\0" * (_ALIGN - len(c) % _ALIGN))


def read_axm_index(path: str):
    """只读头 + 索引, 不映射数据。返回 (index_dict, data_start)。"""
    with open(path, "rb") as f:
        head = f.read(_HEADER.size)
        if len(head) < _HEADER.size:
            raise ValueError(f"{path} 不是 .axm 文件 (文件过短)")
        magic, version, _, idx_len = _HEADER.unpack(head)
        if magic != MAGIC:
            raise ValueError(f"{path} 不是 .axm 文件 (magic={magic!r})")
        if version != VERSION:
            raise ValueError(f".axm 版本不支持: {version}")
        idx = json.loads(f.read(idx_len).decode("utf-8"))
    data_start = _align_up(_HEADER.size + idx_len)
    return idx, data_start


def load_axm(path: str, mmap: bool = True) -> dict:
    """加载整个 .axm → {name: ndarray (memmap 视图)}。

    mmap=True 时零拷贝映射, 不触发实际读取 (首次访问按页调入);
    内存充足时可直接用 (读取由 OS page cache 承担)。
    """
    idx, data_start = read_axm_index(path)
    mode = "r" if mmap else None
    out = {}
    for t in idx["tensors"]:
        arr = np.memmap(path, dtype=t["dtype"], mode=mode,
                        offset=data_start + t["offset"],
                        shape=tuple(t["shape"]))
        out[t["name"]] = arr
    return out


class AxmReader:
    """流式访问单个张量 (适合大模型逐个加载)。"""

    def __init__(self, path: str):
        self.path = path
        self.index, self.data_start = read_axm_index(path)
        self._tensors = {t["name"]: t for t in self.index["tensors"]}

    def names(self):
        return list(self._tensors)

    def meta(self):
        return self.index.get("meta", {})

    def get(self, name: str) -> np.ndarray:
        """读取单个张量 (memmap 视图)。"""
        t = self._tensors[name]
        return np.memmap(self.path, dtype=t["dtype"], mode="r",
                         offset=self.data_start + t["offset"],
                         shape=tuple(t["shape"]))
