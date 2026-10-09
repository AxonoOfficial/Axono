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
"""逐元素算子 (sub/mul/div/neg/abs/exp/log/sqrt/sigmoid/tanh) 测试。"""

import numpy as np
import pytest

import axono


@pytest.mark.parametrize(
    "op,ref",
    [
        (axono.sub, lambda a, b: a - b),
        (axono.mul, lambda a, b: a * b),
        (axono.div, lambda a, b: a / b),
    ],
)
def test_binary_ops_cpu(op, ref):
    a = np.array([[1.5, -2.0, 3.0], [0.0, 4.0, -0.5]], dtype=np.float32)
    b = np.array([[0.5, 4.0, -1.0], [2.0, -1.0, 0.25]], dtype=np.float32)
    ta = axono.Tensor.from_numpy(a)
    tb = axono.Tensor.from_numpy(b)
    out = op(ta, tb)
    assert isinstance(out, axono.Tensor)
    np.testing.assert_allclose(out.to_numpy(), ref(a, b), rtol=1e-6, atol=1e-7)
    # 输入不变
    np.testing.assert_allclose(ta.to_numpy(), a, rtol=1e-7)


def test_unary_ops_cpu():
    a = np.array([1.5, -2.0, 3.0], dtype=np.float32)
    t = axono.Tensor.from_numpy(a)
    np.testing.assert_allclose((-t).to_numpy(), -a, rtol=1e-7)
    np.testing.assert_allclose(abs(t).to_numpy(), np.abs(a), rtol=1e-7)
    np.testing.assert_allclose(axono.exp(t).to_numpy(), np.exp(a), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        axono.sigmoid(t).to_numpy(), 1 / (1 + np.exp(-a)), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        axono.tanh(t).to_numpy(), np.tanh(a), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        axono.sqrt(abs(t)).to_numpy(), np.sqrt(np.abs(a)), rtol=1e-6, atol=1e-7
    )
    pos = np.array([0.5, 2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(
        axono.log(axono.Tensor.from_numpy(pos)).to_numpy(),
        np.log(pos),
        rtol=1e-6,
        atol=1e-7,
    )
    tp = axono.Tensor.from_numpy(np.abs(a) + 0.5)
    np.testing.assert_allclose(
        axono.rsqrt(tp).to_numpy(), 1 / np.sqrt(np.abs(a) + 0.5), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(axono.square(t).to_numpy(), a * a, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        axono.reciprocal(tp).to_numpy(), 1 / (np.abs(a) + 0.5), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(axono.sign(t).to_numpy(), np.sign(a), rtol=1e-7)
    np.testing.assert_allclose(axono.sin(t).to_numpy(), np.sin(a), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(axono.cos(t).to_numpy(), np.cos(a), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(axono.floor(t).to_numpy(), np.floor(a), rtol=1e-7)
    np.testing.assert_allclose(axono.ceil(t).to_numpy(), np.ceil(a), rtol=1e-7)
    np.testing.assert_allclose(axono.round(t).to_numpy(), np.round(a), rtol=1e-7)
    np.testing.assert_allclose(
        axono.pow(tp, tp).to_numpy(),
        (np.abs(a) + 0.5) ** (np.abs(a) + 0.5),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        axono.maximum(t, tp).to_numpy(), np.maximum(a, np.abs(a) + 0.5), rtol=1e-7
    )
    np.testing.assert_allclose(
        axono.minimum(t, tp).to_numpy(), np.minimum(a, np.abs(a) + 0.5), rtol=1e-7
    )


def test_binary_op_shape_mismatch():
    a = axono.Tensor.ones([2, 3])
    b = axono.Tensor.ones([3, 2])
    with pytest.raises(ValueError):
        axono.mul(a, b)


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
@pytest.mark.parametrize(
    "op,ref",
    [
        (axono.sub, lambda a, b: a - b),
        (axono.mul, lambda a, b: a * b),
        (axono.div, lambda a, b: a / b),
        (axono.exp, lambda a: np.exp(a)),
        (axono.sigmoid, lambda a: 1 / (1 + np.exp(-a))),
        (axono.tanh, lambda a: np.tanh(a)),
        (axono.sin, lambda a: np.sin(a)),
        (axono.cos, lambda a: np.cos(a)),
        (axono.square, lambda a: a * a),
        (axono.sign, lambda a: np.sign(a)),
        (axono.floor, lambda a: np.floor(a)),
        (axono.ceil, lambda a: np.ceil(a)),
        (axono.round, lambda a: np.round(a)),
    ],
)
def test_elementwise_cuda(op, ref):
    rng = np.random.default_rng(0)
    if op in (axono.sub, axono.mul, axono.div, axono.pow, axono.maximum, axono.minimum):
        a = rng.standard_normal((16, 16)).astype(np.float32)
        b = rng.standard_normal((16, 16)).astype(np.float32) + 0.5
        ta = axono.Tensor.from_numpy(a).to("cuda")
        tb = axono.Tensor.from_numpy(b).to("cuda")
        out = op(ta, tb)
        np.testing.assert_allclose(
            out.to("cpu").to_numpy(), ref(a, b), rtol=1e-5, atol=1e-6
        )
    else:
        a = rng.standard_normal((16, 16)).astype(np.float32)
        ta = axono.Tensor.from_numpy(a).to("cuda")
        out = op(ta)
        np.testing.assert_allclose(
            out.to("cpu").to_numpy(), ref(a), rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_rsqrt_reciprocal_cuda():
    """rsqrt/reciprocal 需正输入。"""
    rng = np.random.default_rng(2)
    a = (np.abs(rng.standard_normal((16, 16))) + 0.5).astype(np.float32)
    ta = axono.Tensor.from_numpy(a).to("cuda")
    np.testing.assert_allclose(
        axono.rsqrt(ta).to("cpu").to_numpy(), 1 / np.sqrt(a), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        axono.reciprocal(ta).to("cpu").to_numpy(), 1 / a, rtol=1e-5, atol=1e-6
    )


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
@pytest.mark.parametrize(
    "op,ref",
    [
        (axono.pow, lambda a, b: a**b),
        (axono.maximum, lambda a, b: np.maximum(a, b)),
        (axono.minimum, lambda a, b: np.minimum(a, b)),
    ],
)
def test_binary_ops_cuda(op, ref):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)
    ta = axono.Tensor.from_numpy(a).to("cuda")
    tb = axono.Tensor.from_numpy(b).to("cuda")
    out = op(ta, tb)
    np.testing.assert_allclose(
        out.to("cpu").to_numpy(), ref(a, b), rtol=1e-5, atol=1e-6
    )


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_multiple_outputs():
    """多输出: capture 返回 tuple, replay 后所有输出缓冲同时更新。"""
    axono.set_backend("cuda")
    try:
        a = axono.Tensor.randn([32, 32])
        b = axono.Tensor.randn([32, 32])
        a_np = a.to("cpu").to_numpy()
        b_np = b.to("cpu").to_numpy()

        g = axono.CUDAGraph()
        g.capture(lambda: (axono.relu(axono.matmul(a, b)), axono.tanh(axono.sub(a, b))))
        g.replay()
        g.sync()
        o1, o2 = g.outputs
        assert len(g.outputs) == 2
        assert g.output is o1
        np.testing.assert_allclose(
            o1.to("cpu").to_numpy(), np.maximum(a_np @ b_np, 0.0), rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            o2.to("cpu").to_numpy(), np.tanh(a_np - b_np), rtol=1e-4, atol=1e-4
        )

        # 修改输入再回放: 两个输出都应更新
        b.fill(0.5)
        g.replay()
        g.sync()
        np.testing.assert_allclose(
            o2.to("cpu").to_numpy(), np.tanh(a_np - 0.5), rtol=1e-4, atol=1e-4
        )
        g.reset()
    finally:
        axono.set_backend("cpu")


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_graph_context_multi_output():
    """with 语法多输出: g.output = (y1, y2)。"""
    axono.set_backend("cuda")
    try:
        a = axono.Tensor.randn([16, 16])
        b = axono.Tensor.randn([16, 16])
        with axono.cuda_graph() as g:
            y1 = axono.relu(a)
            y2 = axono.sigmoid(b)
            g.output = (y1, y2)
        g.replay()
        g.sync()
        y1_out, y2_out = g.outputs
        a_np = a.to("cpu").to_numpy()
        b_np = b.to("cpu").to_numpy()
        np.testing.assert_allclose(
            y1_out.to("cpu").to_numpy(), np.maximum(a_np, 0.0), rtol=1e-6
        )
        np.testing.assert_allclose(
            y2_out.to("cpu").to_numpy(), 1 / (1 + np.exp(-b_np)), rtol=1e-5
        )
        g.reset()
    finally:
        axono.set_backend("cpu")
