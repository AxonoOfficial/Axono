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
"""Axono vs PyTorch 精度对比。

对每个算子在 CPU 与 CUDA 上各生成若干组随机输入, 分别用 axono 与
torch 计算并逐元素对比 (float32), 输出最大绝对/相对误差。全部算子
应在 float32 机器精度内一致 (最大绝对误差 ~1e-7 量级, 相对误差 <1e-5)。

用法: 在 V100 上  python benchmarks/accuracy_vs_torch.py
"""

from __future__ import annotations

import numpy as np
import torch

import axono

RTOL = 1e-4  # float32 逐元素算子应达 1e-4 相对精度
ATOL = 1e-5


def _max_err(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """返回 (最大绝对误差, 最大相对误差)。"""
    abs_err = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-6)
    return float(abs_err.max()), float((abs_err / denom).max())


def _check(name: str, got: axono.Tensor, ref: np.ndarray, results: list) -> None:
    g = got.to("cpu").to_numpy().astype(np.float64)
    r = ref.astype(np.float64)
    ae, re_ = _max_err(g, r)
    ok = np.allclose(g, r, rtol=RTOL, atol=ATOL)
    results.append((name, ok, ae, re_))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:32s} max_abs={ae:.3e}  max_rel={re_:.3e}")


def main() -> int:
    print("=" * 72)
    print("Axono vs PyTorch float32 精度对比")
    print("=" * 72)
    results: list[tuple[str, bool, float, float]] = []
    rng = np.random.default_rng(42)
    shape = (64, 64)

    for device in ("cpu", "cuda"):
        if device == "cuda" and not axono.cuda_available():
            print("\n[跳过 CUDA: 无可用构建]")
            continue
        if device == "cuda" and not torch.cuda.is_available():
            print("\n[跳过 CUDA: torch 无 GPU]")
            continue
        axono.set_backend(device)
        td = device if device != "cpu" else "cpu"
        print(f"\n----- 设备: {device} -----")

        a = rng.standard_normal(shape).astype(np.float32)
        b = rng.standard_normal(shape).astype(np.float32) + 0.3
        b_pos = np.abs(b) + 0.5  # 保证 >0 (log/sqrt/div 用)
        ta = axono.Tensor.from_numpy(a).to(device)
        tb = axono.Tensor.from_numpy(b).to(device)
        tbp = axono.Tensor.from_numpy(b_pos).to(device)
        tha = torch.from_numpy(a).to(td)
        thb = torch.from_numpy(b).to(td)
        thbp = torch.from_numpy(b_pos).to(td)

        def got(t: axono.Tensor) -> np.ndarray:
            return t.to("cpu").to_numpy()

        def ref(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        # 二元算子
        _check("add", axono.add(ta, tb), ref(tha + thb), results)
        _check("sub", axono.sub(ta, tb), ref(tha - thb), results)
        _check("mul", axono.mul(ta, tb), ref(tha * thb), results)
        _check("div", axono.div(ta, tbp), ref(tha / thbp), results)
        _check("matmul", axono.matmul(ta, tb), ref(tha @ thb), results)

        # 一元算子
        _check("neg", axono.neg(ta), ref(-tha), results)
        _check("abs", axono.abs(ta), ref(torch.abs(tha)), results)
        _check("exp", axono.exp(ta), ref(torch.exp(tha)), results)
        _check("log", axono.log(tbp), ref(torch.log(thbp)), results)
        _check("sqrt", axono.sqrt(tbp), ref(torch.sqrt(thbp)), results)
        _check("sigmoid", axono.sigmoid(ta), ref(torch.sigmoid(tha)), results)
        _check("tanh", axono.tanh(ta), ref(torch.tanh(tha)), results)
        _check("relu", axono.relu(ta), ref(torch.relu(tha)), results)

        # CUDA Graph 回放精度: 同一计算, eager vs replay
        if device == "cuda":
            print("  -- CUDA Graph 回放精度 --")
            m1 = axono.matmul(ta, tb)
            with axono.cuda_graph() as g:
                y = axono.relu(axono.matmul(ta, tb))
                g.output = y
            g.replay()
            g.sync()
            _check("graph: relu(matmul)", g.output,
                   ref(torch.relu(tha @ thb)), results)

    axono.set_backend("cpu")
    n_pass = sum(1 for r in results if r[1])
    print("\n" + "=" * 72)
    print(f"结果: {n_pass}/{len(results)} 通过  (rtol={RTOL}, atol={ATOL})")
    if n_pass != len(results):
        for name, ok, ae, re_ in results:
            if not ok:
                print(f"  FAIL: {name}  max_abs={ae:.3e}  max_rel={re_:.3e}")
        return 1
    print("全部算子与 PyTorch 精度一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
