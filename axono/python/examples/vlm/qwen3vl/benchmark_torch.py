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
"""Qwen3-VL-2B 前向性能对比: Axono vs HuggingFace transformers。

用法:
    PYTHONPATH=. python -m examples.vlm.qwen3vl.benchmark_torch \
        --device cuda --model /path/to/Qwen3-VL-2B-Instruct

分别测 Axono 与 torch 的单次全前向 (视觉塔 + 文本塔 + lm_head),
min-of-N 取稳定值并打印加速比。可选 --plot 输出 PNG 柱状图。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import axono

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.vlm.qwen3vl.model import (  # noqa: E402
    IMAGE_TOKEN_ID,
    Qwen3VLForConditionalGeneration,
)

DEFAULT_MODEL = "/root/autodl-tmp/qwen3vl-2b"


def make_inputs(model_dir: str, grid_h: int, grid_w: int):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    v = cfg["vision_config"]
    patch = v["patch_size"]
    n = grid_h * grid_w
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal(
        (n, 3, v["temporal_patch_size"], patch, patch)
    ).astype(np.float32)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    want = n // (v["spatial_merge_size"] ** 2)
    ids = (
        tok.encode("<|im_start|>user\n").ids
        + [IMAGE_TOKEN_ID] * want
        + tok.encode("描述这张图片<|im_end|>\n<|im_start|>assistant\n").ids
    )
    return pixels, ids


def bench(fn, warmup: int = 3, reps: int = 5, trials: int = 5) -> float:
    """返回单次调用的最优耗时 (ms)。"""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        best = min(best, (time.time() - t0) / reps * 1000)
    return best


def run_axono(model_dir, device, pixels, ids, grid_h, grid_w):
    axono.set_backend(device)
    m = Qwen3VLForConditionalGeneration(
        os.path.join(model_dir, "config.json"), device=device
    )
    m.load_hf_weights(model_dir)
    fn = lambda: m(pixels, 1, grid_h, grid_w, ids)  # noqa: E731
    return bench(fn)


def run_torch(model_dir, device, pixels, ids, grid_h, grid_w):
    import torch
    from transformers.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration as Qwen3VLHF,
)

    if device != "cuda":
        raise SystemExit("torch 基线仅支持 cuda")
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    n = grid_h * grid_w
    v = cfg["vision_config"]
    ref = Qwen3VLHF.from_pretrained(model_dir, torch_dtype=torch.float32).to(device).eval()
    pv = torch.from_numpy(
        pixels.reshape(n, 3 * v["temporal_patch_size"], v["patch_size"], v["patch_size"])
    ).to(device)
    gthw = torch.tensor([[1, grid_h, grid_w]], device=device)
    iid = torch.tensor([ids], device=device)
    mm = torch.zeros_like(iid)
    mm[iid == IMAGE_TOKEN_ID] = 1

    def fn():
        with torch.no_grad():
            return ref(
                input_ids=iid,
                pixel_values=pv,
                image_grid_thw=gthw,
                mm_token_type_ids=mm,
            )

    return bench(fn)


def plot_bars(results: dict, out_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    vals = [results[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    bars = ax.bar(labels, vals, color=["#8B5CF6", "#F59E0B"][: len(labels)])
    ax.set_ylabel("forward latency (ms)")
    ax.set_title("Qwen3-VL-2B full forward (V100, fp32)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
    if len(vals) == 2 and min(vals) > 0:
        ax.set_xlabel(f"speedup {max(vals) / min(vals):.2f}x")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"图已保存: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--grid", type=int, default=16, help="视觉网格边长 (grid_h=grid_w)")
    ap.add_argument("--plot", default=None, help="输出 PNG 路径")
    args = ap.parse_args()

    pixels, ids = make_inputs(args.model, args.grid, args.grid)
    print(f"输入: {args.grid}x{args.grid} patch, prompt {len(ids)} tokens, device={args.device}")

    results = {}
    results["Axono"] = run_axono(args.model, args.device, pixels, ids, args.grid, args.grid)
    print(f"Axono: {results['Axono']:.1f} ms")
    if args.device == "cuda":
        results["torch"] = run_torch(args.model, args.device, pixels, ids, args.grid, args.grid)
        print(f"torch: {results['torch']:.1f} ms")
        print(f"加速比 (torch / Axono): {results['torch'] / results['Axono']:.2f}x")

    if args.plot:
        plot_bars(results, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
