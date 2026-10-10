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
"""序列/LLM 结构算子测试: embedding/rope/attention/concat/slice/argmax。"""

import math

import numpy as np
import pytest

import axono
from axono import DataType, Tensor

np.random.seed(2027)


def _ids_tensor(arr, device):
    t = Tensor.zeros(tuple(np.asarray(arr).shape), dtype=DataType.INT64,
                     device=device)
    t.copy_from_numpy(np.asarray(arr, dtype=np.int64))
    return t


def _ref_attention(q, k, v, causal):
    sq, hq, d = q.shape
    skv, hkv = k.shape[:2]
    group = hq // hkv
    out = np.zeros((sq, hq, d), np.float32)
    scale = 1.0 / math.sqrt(d)
    for t in range(sq):
        for h in range(hq):
            hk = h // group
            kv_end = min(t + 1, skv) if causal else skv
            scores = q[t, h] @ k[:kv_end, hk].T * scale
            scores = scores - scores.max()
            p = np.exp(scores)
            p /= p.sum()
            out[t, h] = p @ v[:kv_end, hk]
    return out


# ---------------- embedding ----------------


def test_embedding_cpu():
    table = np.random.randn(10, 4).astype(np.float32)
    ids = np.array([[1, 3], [5, 2]], dtype=np.int64)
    out = axono.embedding(_ids_tensor(ids, "cpu"), Tensor.from_numpy(table))
    np.testing.assert_allclose(out.to_numpy(), table[ids], rtol=1e-6)


# ---------------- rope ----------------


def test_rope_cpu():
    seq, heads, dim = 4, 2, 8
    x = np.random.randn(seq, heads, dim).astype(np.float32)
    pos = np.arange(seq, dtype=np.int64)
    theta = 10000.0
    out = axono.rope(Tensor.from_numpy(x), _ids_tensor(pos, "cpu"), theta)
    # 参考实现 (HF rotate_half half-split: (i, i+half) 配对)
    ref = np.empty_like(x)
    half = dim // 2
    inv = theta ** (-2.0 * np.arange(0, half, dtype=np.float32) / dim)
    for t in range(seq):
        ang = pos[t] * inv
        c, s = np.cos(ang), np.sin(ang)
        for h in range(heads):
            x0, x1 = x[t, h, :half], x[t, h, half:]
            ref[t, h, :half] = x0 * c - x1 * s
            ref[t, h, half:] = x1 * c + x0 * s
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-4, atol=1e-5)


# ---------------- attention ----------------


@pytest.mark.parametrize("causal", [False, True])
def test_attention_cpu(causal):
    sq, skv, hq, hkv, d = 4, 6, 4, 2, 8
    q = np.random.randn(sq, hq, d).astype(np.float32)
    k = np.random.randn(skv, hkv, d).astype(np.float32)
    v = np.random.randn(skv, hkv, d).astype(np.float32)
    out = axono.scaled_dot_product_attention(
        Tensor.from_numpy(q), Tensor.from_numpy(k), Tensor.from_numpy(v), causal
    )
    np.testing.assert_allclose(
        out.to_numpy(), _ref_attention(q, k, v, causal), rtol=1e-4, atol=1e-5
    )


def test_attention_prefill_full_causal_cpu():
    """prefill (skv == sq) 因果掩码下每步只见过去"""
    sq, hq, d = 6, 2, 8
    hkv = 2
    q = np.random.randn(sq, hq, d).astype(np.float32)
    k = np.random.randn(sq, hkv, d).astype(np.float32)
    v = np.random.randn(sq, hkv, d).astype(np.float32)
    out = axono.scaled_dot_product_attention(
        Tensor.from_numpy(q), Tensor.from_numpy(k), Tensor.from_numpy(v), True
    )
    np.testing.assert_allclose(
        out.to_numpy(), _ref_attention(q, k, v, True), rtol=1e-4, atol=1e-5
    )


# ---------------- concat / slice ----------------


def test_concat_last_dim_cpu():
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 3, 5).astype(np.float32)
    out = axono.concat(Tensor.from_numpy(a), Tensor.from_numpy(b), -1)
    np.testing.assert_allclose(out.to_numpy(), np.concatenate([a, b], -1), rtol=1e-6)


def test_concat_axis0_cpu():
    a = np.random.randn(2, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    out = axono.concat(Tensor.from_numpy(a), Tensor.from_numpy(b), 0)
    np.testing.assert_allclose(out.to_numpy(), np.concatenate([a, b], 0), rtol=1e-6)


def test_slice_last_dim_cpu():
    x = np.random.randn(3, 8).astype(np.float32)
    out = axono.slice_(Tensor.from_numpy(x), -1, 2, 4)
    np.testing.assert_allclose(out.to_numpy(), x[:, 2:6], rtol=1e-6)


# ---------------- argmax ----------------


def test_argmax_cpu():
    x = np.random.randn(5, 16).astype(np.float32)
    out = axono.argmax(Tensor.from_numpy(x))
    assert out.dtype == DataType.INT64
    np.testing.assert_array_equal(out.to_numpy(), x.argmax(-1))


# ---------------- CUDA ----------------


def test_embedding_cuda(cuda_env):
    table = Tensor.randn((64, 32), device="cuda")
    ids = _ids_tensor(np.random.randint(0, 64, (2, 5)), "cuda")
    out = axono.embedding(ids, table)
    tn = table.to_numpy()
    np.testing.assert_allclose(out.to_numpy(), tn[ids.to_numpy()], rtol=1e-5)


def test_rope_cuda(cuda_env):
    x = Tensor.randn((8, 4, 32), device="cuda")
    pos = _ids_tensor(np.arange(8), "cuda")
    out = axono.rope(x, pos, 10000.0)
    xn = x.to_numpy()
    ref = np.empty_like(xn)
    dim = 32
    half = dim // 2
    inv = 10000.0 ** (-2.0 * np.arange(0, half, dtype=np.float32) / dim)
    for t in range(8):
        ang = t * inv
        c, s = np.cos(ang), np.sin(ang)
        for h in range(4):
            x0, x1 = xn[t, h, :half], xn[t, h, half:]
            ref[t, h, :half] = x0 * c - x1 * s
            ref[t, h, half:] = x1 * c + x0 * s
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_cuda(cuda_env, causal):
    sq, skv, hq, hkv, d = 8, 12, 4, 2, 64
    q = Tensor.randn((sq, hq, d), device="cuda")
    k = Tensor.randn((skv, hkv, d), device="cuda")
    v = Tensor.randn((skv, hkv, d), device="cuda")
    out = axono.scaled_dot_product_attention(q, k, v, causal)
    ref = _ref_attention(q.to_numpy(), k.to_numpy(), v.to_numpy(), causal)
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-3, atol=1e-3)


def test_concat_slice_cuda(cuda_env):
    a = Tensor.randn((4, 8), device="cuda")
    b = Tensor.randn((4, 6), device="cuda")
    c = axono.concat(a, b, -1).to_numpy()
    np.testing.assert_allclose(
        c, np.concatenate([a.to_numpy(), b.to_numpy()], -1), rtol=1e-5
    )
    s = axono.slice_(a, -1, 1, 5).to_numpy()
    np.testing.assert_allclose(s, a.to_numpy()[:, 1:6], rtol=1e-5)


def test_argmax_cuda(cuda_env):
    x = Tensor.randn((7, 128), device="cuda")
    out = axono.argmax(x)
    np.testing.assert_array_equal(out.to_numpy(), x.to_numpy().argmax(-1))


def test_full_text_block_cuda(cuda_env):
    """一个完整 transformer block: linear+rmsnorm+rope+attention+mlp 端到端"""
    d_model, n_head, n_kv, d_head = 32, 4, 2, 8
    d_ff = 64
    seq = 6
    x = Tensor.randn((seq, d_model), device="cuda")
    wq = Tensor.randn((n_head * d_head, d_model), device="cuda")
    wk = Tensor.randn((n_kv * d_head, d_model), device="cuda")
    wv = Tensor.randn((n_kv * d_head, d_model), device="cuda")
    wo = Tensor.randn((d_model, d_model), device="cuda")
    w_gate = Tensor.randn((d_ff, d_model), device="cuda")
    w_up = Tensor.randn((d_ff, d_model), device="cuda")
    w_down = Tensor.randn((d_model, d_ff), device="cuda")
    rn_w = Tensor.ones((d_model,), dtype=DataType.FLOAT32, device="cuda")

    pos = _ids_tensor(np.arange(seq), "cuda")
    # attention
    h = axono.rms_norm(x, rn_w)
    q = axono.linear_nobias(h, wq).to_numpy().reshape(seq, n_head, d_head)
    k = axono.linear_nobias(h, wk).to_numpy().reshape(seq, n_kv, d_head)
    v = axono.linear_nobias(h, wv).to_numpy().reshape(seq, n_kv, d_head)
    tq = Tensor.from_numpy(q).to("cuda")
    tk = Tensor.from_numpy(k).to("cuda")
    q = axono.rope(tq, pos, 10000.0).to_numpy()
    k = axono.rope(tk, pos, 10000.0).to_numpy()
    attn = axono.scaled_dot_product_attention(
        Tensor.from_numpy(q).to("cuda"),
        Tensor.from_numpy(k).to("cuda"),
        Tensor.from_numpy(v).to("cuda"),
        True,
    )
    attn_out = axono.linear_nobias(
        Tensor.from_numpy(attn.to_numpy().reshape(seq, n_head * d_head)).to("cuda"), wo
    )
    # mlp (SwiGLU)
    g = axono.linear_nobias(h, w_gate)
    u = axono.linear_nobias(h, w_up)
    mlp = axono.linear_nobias(axono.mul(g, axono.silu(u)), w_down)
    # 残差
    y = axono.add(axono.add(x, attn_out), mlp)
    assert list(y.shape) == [seq, d_model]
    assert np.isfinite(y.to_numpy()).all()


# ---------------------------------------------------------------------------
# mrope_cos_sin
# ---------------------------------------------------------------------------
def _ref_mrope_cos_sin(pos3, theta, d_head, h_sec, w_sec):
    inv = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    freqs = pos3[:, :, None].astype(np.float32) * inv[None, None, :]
    out = freqs[0].copy()
    out[:, np.arange(1, h_sec * 3, 3)] = freqs[1][:, np.arange(1, h_sec * 3, 3)]
    out[:, np.arange(2, w_sec * 3, 3)] = freqs[2][:, np.arange(2, w_sec * 3, 3)]
    angles = np.concatenate([out, out], axis=-1)
    return np.cos(angles), np.sin(angles)


def test_mrope_cos_sin_cpu():
    rng = np.random.default_rng(7)
    seq, d_head = 33, 128
    pos3 = rng.integers(0, 50, size=(3, seq)).astype(np.int64)
    theta = 5e6
    inv = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    cos_t, sin_t = axono.mrope_cos_sin(
        _ids_tensor(pos3, "cpu"),
        Tensor.from_numpy(inv.astype(np.float32)), 20, 20)
    rc, rs = _ref_mrope_cos_sin(pos3, theta, d_head, 20, 20)
    np.testing.assert_allclose(cos_t.to_numpy(), rc, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sin_t.to_numpy(), rs, rtol=1e-5, atol=1e-6)


def test_mrope_cos_sin_cuda(cuda_env):
    rng = np.random.default_rng(8)
    seq, d_head = 17, 128
    pos3 = rng.integers(0, 50, size=(3, seq)).astype(np.int64)
    theta = 5e6
    inv = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    cos_t, sin_t = axono.mrope_cos_sin(
        _ids_tensor(pos3, "cuda:0"),
        Tensor.from_numpy(inv.astype(np.float32)).to("cuda:0"), 20, 20)
    rc, rs = _ref_mrope_cos_sin(pos3, theta, d_head, 20, 20)
    np.testing.assert_allclose(cos_t.to_numpy(), rc, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sin_t.to_numpy(), rs, rtol=1e-5, atol=1e-6)


def test_to_same_device_no_copy(cuda_env):
    """torch 语义: 同设备 to() 返回自身 (共享 storage)。"""
    x = Tensor.from_numpy(np.ones((4,), np.float32)).to("cuda:0")
    y = x.to("cuda:0")
    assert y is x
    z = x.to("cpu")
    assert z is not x

