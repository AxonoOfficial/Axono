"""axono.benchmarks — 性能基准套件 (随库分发)。

运行::

    python -m axono.benchmarks.bench            # 生成 benchmark_results.json
    python -m axono.benchmarks.plot_benchmark   # 生成合并大图 benchmark.png

bench.py 对比 Axono 与 PyTorch 在 CPU / CUDA 上的 matmul / add / relu 平均耗时。
"""
