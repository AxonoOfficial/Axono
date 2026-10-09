#!/usr/bin/env python3
"""Axono v0.2 性能基准 — 对比 PyTorch (CPU + CUDA)。

随库分发, 运行::

    python -m axono.benchmarks.bench            # 生成 benchmark_results.json
    python -m axono.benchmarks.plot_benchmark   # 合并大图 benchmark.png

每个用例自适应重复次数 (目标单用例 ~0.5s), 避免 CPU 大矩阵拖慢整体。
额外给出全局 inplace 开关开启时的 add 耗时 (展示零分配收益)。
"""

import json
import os
import sys
import time

import numpy as np

# 支持从库内直接运行: python -m axono.benchmarks.bench
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
import axono  # noqa: E402
from axono.core.ops import relu as _axono_relu  # noqa: E402

import torch  # noqa: E402

WARMUP = 5
TARGET_S = 0.5  # 每用例目标测量时长

# (算子, 规模列表)
SIZES = [
    ("matmul", [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]),
    ("add", [(1024, 1024), (4096, 4096)]),
    ("relu", [(1024, 1024), (4096, 4096)]),
]


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _auto_repeat(fn, device):
    """先测单次耗时, 再定重复次数 (上限 50)。"""
    for _ in range(WARMUP):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    fn()
    _sync(device)
    one = time.perf_counter() - t0
    return max(3, min(50, int(TARGET_S / max(one, 1e-6))))


def _timeit(fn, device):
    n = _auto_repeat(fn, device)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n


def bench_axono(op, shape, device, inplace=False):
    axono.set_default_device(device)
    axono.set_inplace_enabled(inplace)
    try:
        if op == "matmul":
            a = axono.Tensor.randn([shape[0], shape[1]])
            b = axono.Tensor.randn([shape[1], shape[0]])
            return _timeit(lambda: axono.operators.matmul(a, b), device)
        if op == "add":
            a = axono.Tensor.randn(list(shape))
            b = axono.Tensor.randn(list(shape))
            return _timeit(lambda: axono.operators.add(a, b), device)
        a = axono.Tensor.randn(list(shape))
        return _timeit(lambda: _axono_relu(a), device)
    finally:
        axono.set_inplace_enabled(False)


def bench_torch(op, shape, device):
    t = torch.device(device)
    if op == "matmul":
        a = torch.randn(shape[0], shape[1], device=t)
        b = torch.randn(shape[1], shape[0], device=t)
        return _timeit(lambda: a @ b, device)
    if op == "add":
        a = torch.randn(*shape, device=t)
        b = torch.randn(*shape, device=t)
        return _timeit(lambda: a + b, device)
    a = torch.randn(*shape, device=t)
    return _timeit(lambda: torch.relu(a), device)


def main():
    results = {}
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for dev in devices:
        print(f"=== device: {dev} ===", flush=True)
        for op, sizes in SIZES:
            for shape in sizes:
                key = f"{op}/{shape[0]}x{shape[1]}/{dev}"
                try:
                    ta = bench_axono(op, shape, dev)
                    tt = bench_torch(op, shape, dev)
                    entry = {"axono_s": ta, "torch_s": tt,
                             "speedup_vs_torch": tt / ta}
                    # inplace 开关场景 (仅 add 同形, 展示零分配路径)
                    if op == "add":
                        entry["axono_inplace_s"] = bench_axono(
                            op, shape, dev, inplace=True)
                    results[key] = entry
                    extra = (f"  inplace={entry['axono_inplace_s']*1e3:8.3f}ms"
                             if "axono_inplace_s" in entry else "")
                    print(f"{key:30s} axono={ta*1e3:9.3f}ms  "
                          f"torch={tt*1e3:9.3f}ms  "
                          f"ratio={tt/ta:6.2f}x{extra}", flush=True)
                except Exception as e:
                    results[key] = {"error": str(e)}
                    print(f"{key:30s} ERROR: {e}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
