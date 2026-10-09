#!/usr/bin/env python3
"""Axono v0.2 性能基准 — 对比 PyTorch (CPU OpenBLAS + CUDA)。

随库分发, 运行::

    python -m axono.benchmarks.bench            # 生成 benchmark_results.json
    python -m axono.benchmarks.plot_benchmark   # 合并大图 benchmark.png

每个用例自适应重复次数 (目标单用例 ~0.5s), 避免 CPU 大矩阵拖慢整体。
CPU 仅测 OpenBLAS 后端 (axono CPU 原生路径已由 OpenBLAS 取代, 不再单独
测量); 另测 CUDA Graph 回放 vs eager 的提交开销收益。
"""

import json
import os
import sys
import time

# 支持从库内直接运行: python -m axono.benchmarks.bench
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
import torch  # noqa: E402

import axono  # noqa: E402
from axono.core.ops import relu as _axono_relu  # noqa: E402

WARMUP = 5
TARGET_S = 0.5  # 每用例目标测量时长

# (算子, 规模列表)
SIZES = [
    ("matmul", [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]),
    ("add", [(1024, 1024), (4096, 4096)]),
    ("relu", [(1024, 1024), (4096, 4096)]),
]

# CUDA Graph 用例: (名称, 规模) —— 小规模多算子链, 展示提交开销收益
GRAPH_SIZES = [
    ("mlp", 256),
    ("mlp", 1024),
]


def _device_info():
    """采集设备元信息 (供图中标注)。"""
    info = {}
    try:
        import platform

        info["cpu"] = platform.processor() or platform.machine()
    except Exception:
        info["cpu"] = "unknown"
    info["cpu_threads"] = os.cpu_count()
    info["cpu_blas"] = "OpenBLAS"
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda or ""
        info["gpu_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    else:
        info["gpu"] = None
    return info


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


def bench_axono(op, shape, device):
    axono.set_default_device(device)
    try:
        if op == "matmul":
            a = axono.Tensor.randn([shape[0], shape[1]])
            b = axono.Tensor.randn([shape[1], shape[0]])
            return _timeit(lambda: axono.matmul(a, b), device)
        if op == "add":
            a = axono.Tensor.randn(list(shape))
            b = axono.Tensor.randn(list(shape))
            return _timeit(lambda: axono.add(a, b), device)
        a = axono.Tensor.randn(list(shape))
        return _timeit(lambda: _axono_relu(a), device)
    finally:
        axono.set_default_device("cpu")


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


def _mlp_chain(n):
    """构造 n 层 matmul+relu 链的调用器 (返回闭包与输入)。"""
    a = axono.Tensor.randn([n, n])
    w = axono.Tensor.randn([n, n])
    return lambda: _axono_relu(axono.matmul(a, w)), a, w


def bench_graph(shape_n):
    """CUDA Graph 回放 vs eager: 同一 matmul+relu 链。"""
    axono.set_default_device("cuda:0")
    try:
        fn, a, w = _mlp_chain(shape_n)
        eager = _timeit(fn, "cuda:0")
        g = axono.CUDAGraph()
        g.capture(fn)
        gr = _timeit(lambda: g.replay(), "cuda:0")
        return eager, gr
    finally:
        axono.set_default_device("cpu")


def main():
    results = {"_meta": _device_info()}
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for dev in devices:
        print(f"=== device: {dev} ===", flush=True)
        for op, sizes in SIZES:
            for shape in sizes:
                key = f"{op}/{shape[0]}x{shape[1]}/{dev}"
                try:
                    ta = bench_axono(op, shape, dev)
                    tt = bench_torch(op, shape, dev)
                    results[key] = {
                        "axono_s": ta,
                        "torch_s": tt,
                        "speedup_vs_torch": tt / ta,
                    }
                    print(
                        f"{key:30s} axono={ta * 1e3:9.3f}ms  "
                        f"torch={tt * 1e3:9.3f}ms  "
                        f"ratio={tt / ta:6.2f}x",
                        flush=True,
                    )
                except Exception as e:
                    results[key] = {"error": str(e)}
                    print(f"{key:30s} ERROR: {e}", flush=True)

    # CUDA Graph: eager vs replay (仅 CUDA)
    if torch.cuda.is_available():
        for name, n in GRAPH_SIZES:
            key = f"graph_{name}/{n}x{n}/cuda:0"
            try:
                eager, gr = bench_graph(n)
                results[key] = {"eager_s": eager, "graph_s": gr, "speedup": eager / gr}
                print(
                    f"{key:30s} eager={eager * 1e3:9.3f}ms  "
                    f"graph={gr * 1e3:9.3f}ms  "
                    f"speedup={eager / gr:6.2f}x",
                    flush=True,
                )
            except Exception as e:
                results[key] = {"error": str(e)}
                print(f"{key:30s} ERROR: {e}", flush=True)

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
