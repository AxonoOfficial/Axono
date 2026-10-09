#!/usr/bin/env python3
"""把 benchmark_results.json 渲染成一张合并大图 (PNG)。

运行: python -m axono.benchmarks.plot_benchmark [results.json] [out.png]
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "benchmark_results.json")
_out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_here, "benchmark.png")
with open(_res) as f:
    R = json.load(f)

# 按 device 分面板, 每面板内按算子+尺寸分组柱状图
devices = sorted({k.split("/")[-1] for k in R if "error" not in R[k]})
ops = ["matmul", "add", "relu"]
has_inplace = any("axono_inplace_s" in v for v in R.values())

fig, axes = plt.subplots(1, len(devices), figsize=(7.5 * len(devices), 6))
if len(devices) == 1:
    axes = [axes]

for ax, dev in zip(axes, devices):
    keys = [k for k in R if k.endswith("/" + dev) and "error" not in R[k]]
    keys.sort(key=lambda k: (ops.index(k.split("/")[0])
                             if k.split("/")[0] in ops else 99,
                             int(k.split("/")[1].split("x")[0])))
    labels = ["/".join(k.split("/")[:2]) for k in keys]
    ax_ms = [R[k]["axono_s"] * 1e3 for k in keys]
    torch_ms = [R[k]["torch_s"] * 1e3 for k in keys]
    n = len(keys)
    x = np.arange(n)
    if has_inplace:
        w = 0.27
        groups = [(x - w, ax_ms, "Axono", "#8B5CF6"),
                  (x, torch_ms, "PyTorch", "#F59E0B"),
                  (x + w, [R[k].get("axono_inplace_s", R[k]["axono_s"]) * 1e3
                           for k in keys], "Axono (inplace 开)", "#10B981")]
    else:
        w = 0.38
        groups = [(x - w / 2, ax_ms, "Axono", "#8B5CF6"),
                  (x + w / 2, torch_ms, "PyTorch", "#F59E0B")]
    bars = []
    for off, vals, lab, color in groups:
        bars.append(ax.bar(off, vals, w, label=lab, color=color))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("耗时 (ms, log)")
    ax.set_title(f"{dev.upper()} — Axono vs PyTorch (V100)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    for b in bars:
        for rect in b:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=6)

fig.suptitle("Axono v0.2 重构性能基准 (柱上数值 = 平均耗时 ms)", fontsize=13)
fig.tight_layout()
fig.savefig(_out, dpi=150)
print("saved", _out)
