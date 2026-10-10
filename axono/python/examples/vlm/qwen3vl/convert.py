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
"""HF safetensors 权重 → Axono .axm 格式转换脚本。

用法::

    python -m examples.vlm.qwen3vl.convert \
        --model /root/autodl-tmp/qwen3vl-2b \
        [--dtype float32|float16] [--out model.axm]

- 键名转成 Axono 模型的扁平参数名 (model.visual.* / model.language_model.*);
- patch_embed.proj.weight 按模型要求 reshape (out, 3*patch*patch);
- 默认输出 <model_dir>/model.axm;
- 可选 --dtype float16 减半文件体积 (加载时转回 fp32)。
"""

import argparse
import json
import os
import sys
import time


def convert(model_dir: str, out_path: str, dtype: str | None) -> dict:
    from safetensors.torch import load_file

    from axono.format import save_axm

    t0 = time.perf_counter()
    path_st = os.path.join(model_dir, "model.safetensors")
    raw = load_file(path_st)  # torch 张量 (bf16 原生支持)
    tensors = {}
    for k, v in raw.items():
        arr = v.float().cpu().numpy()  # 统一 fp32 (Axono 运算 dtype)
        if k.endswith("patch_embed.proj.weight"):
            p = arr.shape
            arr = arr.reshape(p[0], -1)
        tensors[k] = arr
    del raw
    t_read = time.perf_counter() - t0

    meta = {
        "source": "hf-safetensors",
        "model_dir": os.path.basename(model_dir),
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": dtype or "keep",
    }
    t1 = time.perf_counter()
    save_axm(tensors, out_path, meta=meta, dtype=dtype)
    t_write = time.perf_counter() - t1

    size = os.path.getsize(out_path) / 1e9
    n = len(tensors)
    del tensors
    return {"n_tensors": n, "size_gb": size, "t_read": t_read,
            "t_write": t_write}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF 模型目录")
    ap.add_argument("--out", default=None, help="输出 .axm 路径 "
                    "(默认 <model>/model.axm)")
    ap.add_argument("--dtype", default=None,
                    choices=["float32", "float16"],
                    help="统一存储 dtype (默认保留源 dtype)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.model, "model.axm")
    st = convert(args.model, out, args.dtype)
    print(f"转换完成: {out}")
    print(f"  张量数 {st['n_tensors']}, 体积 {st['size_gb']:.2f} GB, "
          f"读取 {st['t_read']:.1f}s + 写出 {st['t_write']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
