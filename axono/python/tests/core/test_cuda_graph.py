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
"""CUDA Graph 捕获/回放测试。

关键语义: capture 体内新建的输出张量由图的私有内存池持有, 其地址在
回放间保持不变 —— 因此 Python 层保存 ``fn()`` 的返回值 (g.output),
replay 后从同一缓冲区读结果。
"""

import numpy as np
import pytest

import axono


@pytest.fixture()
def cuda_env():
    if not axono.cuda_available():
        pytest.skip("需要 CUDA 构建")
    axono.set_backend("cuda")
    yield
    axono.set_backend("cpu")


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_matmul_relu(cuda_env):
    a = axono.Tensor.randn([64, 64])
    b = axono.Tensor.randn([64, 64])
    a_np, b_np = a.to("cpu").to_numpy(), b.to("cpu").to_numpy()
    expected = np.maximum(a_np @ b_np, 0.0)

    g = axono.CUDAGraph()
    g.capture(lambda: axono.relu(axono.matmul(a, b)))
    assert g.is_captured
    assert g.num_nodes >= 2  # 至少 matmul + relu
    out = g.output
    assert out is not None and out.is_cuda

    for _ in range(3):
        g.replay()
        g.sync()
        np.testing.assert_allclose(
            out.to("cpu").to_numpy(), expected, rtol=1e-4, atol=1e-4
        )
    g.reset()
    assert not g.is_captured


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_output_updates_with_input(cuda_env):
    """输入数据原地更新 (地址不变) 后, replay 结果随之变化。"""
    a = axono.Tensor.randn([32, 32])
    b = axono.Tensor.full([32, 32], 2.0)

    g = axono.CUDAGraph()
    g.capture(lambda: axono.matmul(a, b))
    out = g.output

    a_np = a.to("cpu").to_numpy()
    rowsum = a_np.sum(axis=1, keepdims=True)  # (32, 1)

    # 先回放一次: a @ (全 2 矩阵) = 行和 * 2 (广播到每行)
    g.replay()
    g.sync()
    np.testing.assert_allclose(
        out.to("cpu").to_numpy(), rowsum * 2.0 + 0.0 * a_np, rtol=1e-3, atol=1e-3
    )

    # 原地把 b 改为全 3 (地址不变), 再回放: 结果应变为 行和 * 3
    b.fill(3.0)
    g.replay()
    g.sync()
    np.testing.assert_allclose(
        out.to("cpu").to_numpy(), rowsum * 3.0 + 0.0 * a_np, rtol=1e-3, atol=1e-3
    )
    g.reset()


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_context_manager(cuda_env):
    """``with axono.cuda_graph() as g:`` 语句块捕获 + 块内 g.output = y。"""
    a = axono.Tensor.randn([16, 16])
    b = axono.Tensor.randn([16, 16])
    a_np, b_np = a.to("cpu").to_numpy(), b.to("cpu").to_numpy()

    with axono.cuda_graph() as g:
        y = axono.relu(axono.matmul(a, b))
        g.output = y  # 显式指定输出张量
    assert g.is_captured
    assert g.num_nodes >= 2

    for _ in range(4):
        g.replay()
        g.sync()
        np.testing.assert_allclose(
            g.output.to("cpu").to_numpy(),
            np.maximum(a_np @ b_np, 0.0),
            rtol=1e-4,
            atol=1e-4,
        )
    g.reset()


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_context_manager_with_temporaries(cuda_env):
    """上下文管理器捕获含临时张量 (图内 alloc 节点) 的链, 可反复回放。

    这是 AutoFreeOnLaunch flag 的关键回归: 不加 flag 时含 alloc 节点的
    图第二次 cudaGraphLaunch 会报 invalid argument。
    """
    a = axono.Tensor.randn([64, 64])
    b = axono.Tensor.randn([64, 64])
    c = axono.Tensor.randn([64, 64])

    with axono.cuda_graph() as g:
        tmp = axono.matmul(a, b)  # 临时: 图内分配
        y = axono.add(axono.relu(tmp), c)
        g.output = y
    assert g.num_nodes >= 3

    ref = None
    for i in range(5):
        g.replay()
        g.sync()
        cur = g.output.to("cpu").to_numpy()
        if ref is None:
            ref = cur
        else:
            np.testing.assert_allclose(cur, ref, rtol=1e-5, atol=1e-5)
    expected = (
        np.maximum(a.to("cpu").to_numpy() @ b.to("cpu").to_numpy(), 0.0)
        + c.to("cpu").to_numpy()
    )
    np.testing.assert_allclose(ref, expected, rtol=1e-3, atol=1e-3)
    g.reset()


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_replay_before_capture_raises(cuda_env):
    g = axono.CUDAGraph()
    with pytest.raises(RuntimeError):
        g.replay()


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_empty_capture_raises(cuda_env):
    """CPU 张量不产生 CUDA kernel, 捕获为空图必须报错。"""
    axono.set_backend("cpu")
    try:
        a = axono.Tensor.randn([4, 4])
        b = axono.Tensor.randn([4, 4])
        g = axono.CUDAGraph()
        with pytest.raises(RuntimeError):
            g.capture(lambda: axono.relu(axono.matmul(a, b)))
    finally:
        axono.set_backend("cuda")  # 恢复 (fixture teardown 期望 cuda)
