#!/usr/bin/env python3
"""把 benchmark_results.json 渲染成统计大图 (PNG), 标注设备信息。

运行: python -m axono.benchmarks.plot_benchmark [results.json] [out.png]

面板:
  1. CPU (OpenBLAS)   — Axono vs PyTorch 各算子耗时
  2. GPU (CUDA)       — 同上
  3. CUDA Graph       — eager vs replay (提交开销收益)
底部标注实测设备 (CPU 型号/线程数/BLAS, GPU 型号/显存/CUDA 版本)。
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体: 优先本仓库自带/系统常见 CJK 字体, 找不到则退回默认 (标签仍可读)
for _fp in (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "NotoSansCJK-Regular.ttc"),
    "/usr/share/fonts/truetype/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

_here = os.path.dirname(os.path.abspath(__file__))
_res = (
    sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "benchmark_results.json")
)
_out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_here, "benchmark.png")

with open(_res) as f:
    R = json.load(f)
meta = R.pop("_meta", {})

AX_COLOR = "#8B5CF6"  # Axono
PT_COLOR = "#F59E0B"  # PyTorch

dev_ops = [
    k for k in R if "/" in k and not k.startswith("graph_") and "error" not in R[k]
]
graph_keys = [k for k in R if k.startswith("graph_") and "error" not in R[k]]

devices = sorted({k.split("/")[-1] for k in dev_ops})
ops = ["matmul", "add", "relu"]


def _sort_key(k):
    p = k.split("/")
    return (ops.index(p[0]) if p[0] in ops else 99, int(p[1].split("x")[0]))


def _panel_bars(ax, dev):
    keys = [k for k in dev_ops if k.endswith("/" + dev)]
    keys.sort(key=_sort_key)
    if not keys:
        ax.axis("off")
        return
    labels = ["/".join(k.split("/")[:2]) for k in keys]
    ax_ms = [R[k]["axono_s"] * 1e3 for k in keys]
    pt_ms = [R[k]["torch_s"] * 1e3 for k in keys]
    ratios = [R[k]["speedup_vs_torch"] for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    b1 = ax.bar(x - w / 2, ax_ms, w, label="Axono", color=AX_COLOR)
    b2 = ax.bar(x + w / 2, pt_ms, w, label="PyTorch", color=PT_COLOR)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("耗时 (ms, log)")
    ax.grid(axis="y", alpha=0.3, which="both")
    for bars, vals in ((b1, ax_ms), (b2, pt_ms)):
        for rect, v in zip(bars, vals):
            ax.annotate(
                f"{v:.2f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
    # 顶部标 Axono 相对 PyTorch 的加速比 (>1 表示 Axono 更快)
    for xi, r in zip(x, ratios):
        ax.annotate(
            f"{r:.2f}x",
            (xi, max(ax_ms[list(x).index(xi)], pt_ms[list(x).index(xi)]) * 1.4),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#10B981",
            fontweight="bold",
        )
    ax.legend(loc="upper left", fontsize=8)


# --- 组装 ---
ncols = len(devices) + (1 if graph_keys else 0)
ncols = max(1, ncols)
fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 6.2))
if ncols == 1:
    axes = [axes]

col = 0
for dev in devices:
    _panel_bars(axes[col], dev)
    title = {"cpu": "CPU (OpenBLAS)", "cuda:0": "GPU (CUDA)"}.get(dev, dev.upper())
    axes[col].set_title(f"{title} — Axono vs PyTorch (V100)")
    col += 1

if graph_keys:
    axg = axes[col]
    gk = sorted(graph_keys)
    labels = [
        k.split("/")[0].replace("graph_", "") + "\n" + k.split("/")[1] for k in gk
    ]
    eager = [R[k]["eager_s"] * 1e3 for k in gk]
    grepl = [R[k]["graph_s"] * 1e3 for k in gk]
    sp = [R[k]["speedup"] for k in gk]
    x = np.arange(len(gk))
    w = 0.38
    be = axg.bar(x - w / 2, eager, w, label="eager", color="#EF4444")
    bg = axg.bar(x + w / 2, grepl, w, label="CUDA Graph replay", color="#10B981")
    for bars, vals in ((be, eager), (bg, grepl)):
        for rect, v in zip(bars, vals):
            axg.annotate(
                f"{v:.3f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for xi, s in zip(x, sp):
        axg.annotate(
            f"{s:.1f}x",
            (xi, max(eager[list(x).index(xi)], grepl[list(x).index(xi)]) * 1.3),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#10B981",
            fontweight="bold",
        )
    axg.set_xticks(x)
    axg.set_xticklabels(labels, fontsize=8)
    axg.set_ylabel("耗时 (ms)")
    axg.set_title("CUDA Graph — replay vs eager (提交开销收益)")
    axg.grid(axis="y", alpha=0.3)
    axg.legend(loc="upper left", fontsize=8)

fig.suptitle(
    "Axono v0.2 重构性能基准 (柱上数值 = 平均耗时 ms; 绿色 = Axono 加速倍率)",
    fontsize=13,
    y=0.98,
)

# 设备信息脚注
parts = []
if meta.get("cpu"):
    parts.append(
        f"CPU: {meta['cpu']} ({meta.get('cpu_threads', '?')} 线程, "
        f"{meta.get('cpu_blas', 'OpenBLAS')})"
    )
if meta.get("gpu"):
    parts.append(
        f"GPU: {meta['gpu']} ({meta.get('gpu_mem_gb', '?')} GB, "
        f"CUDA {meta.get('cuda', '?')})"
    )
if parts:
    fig.text(0.5, 0.005, "  |  ".join(parts), ha="center", fontsize=9, color="#555555")

fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(_out, dpi=150)
print("saved", _out)
