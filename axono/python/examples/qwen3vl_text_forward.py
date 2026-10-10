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
"""Qwen3-VL-2B 文本前向 (纯 Axono 算子): 加载 safetensors 权重,
跑 prefill + greedy 解码, 对比 transformers 参考 logits。

用法: PYTHONPATH=. python examples/qwen3vl_text_forward.py \
        --model /root/autodl-tmp/qwen3vl-2b --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import axono

PROMPT = "中国的首都是哪里？"


def load_weights(path: str, device: str) -> dict:
    """safetensors (bf16) -> fp32 axono Tensor dict"""
    from safetensors.torch import load_file

    raw = load_file(path)
    w = {}
    for k, v in raw.items():
        arr = v.float().cpu().numpy()
        t = axono.Tensor.from_numpy(np.ascontiguousarray(arr))
        w[k] = t.to(device) if device != "cpu" else t
    return w


def tokenizer_encode(text: str) -> list:
    """占位: 优先使用 tokenizers 库 (见 main)。"""
    raise NotImplementedError


def qwen3_text_forward(
    w: dict,
    ids: list,
    device: str,
    cfg: dict,
    hidden: np.ndarray | None = None,
    deepstack: list | None = None,
    image_positions: np.ndarray | None = None,
    mrope_pos3: np.ndarray | None = None,
) -> np.ndarray:
    """Qwen3 文本主干前向, 返回 logits (numpy, (1, seq, vocab))。

    hidden: 可选预计算 embedding (1, seq, hidden), 替代 ids 的 embedding
            (用于端到端: 已替换 image token 位置为视觉特征)。
    deepstack: 可选 list of ndarray (n_img, hidden) — 加到前 len(deepstack)
            层输出的 image token 位置 (HF _deepstack_process 约定)。
    image_positions: image token 在序列中的位置索引 (int64)。
    """
    t = cfg.get("text_config", cfg)
    n_head = t["num_attention_heads"]
    n_kv = t["num_key_value_heads"]
    d_head = t["head_dim"]
    layers = t["num_hidden_layers"]
    eps = t["rms_norm_eps"]
    theta = float(t["rope_theta"])

    seq = len(ids)
    if hidden is not None:
        h = to_t2(hidden, device)  # (1, seq, hidden)
    else:
        ids_t = _ids(ids, device)
        h = axono.embedding(ids_t, w["model.language_model.embed_tokens.weight"])
    pos = _arange(seq, device)

    for li in range(layers):
        p = f"model.language_model.layers.{li}"
        # --- attention ---
        x = axono.rms_norm(h, w[f"{p}.input_layernorm.weight"], eps)
        q = _reshape3(
            axono.linear_nobias(x, w[f"{p}.self_attn.q_proj.weight"]),
            seq,
            n_head,
            d_head,
            device,
        )
        k = _reshape3(
            axono.linear_nobias(x, w[f"{p}.self_attn.k_proj.weight"]),
            seq,
            n_kv,
            d_head,
            device,
        )
        v = _reshape3(
            axono.linear_nobias(x, w[f"{p}.self_attn.v_proj.weight"]),
            seq,
            n_kv,
            d_head,
            device,
        )
        # Qwen3: q/k norm (head_dim 上 RMSNorm) 在 RoPE 之前
        q = _head_norm(
            q, w[f"{p}.self_attn.q_norm.weight"], eps, seq, n_head, d_head, device
        )
        k = _head_norm(
            k, w[f"{p}.self_attn.k_norm.weight"], eps, seq, n_kv, d_head, device
        )
        if mrope_pos3 is not None:
            mc, ms = mrope_cos_sin(mrope_pos3, theta, d_head)
            q = rope_cs(q, mc, ms, device)
            k = rope_cs(k, mc, ms, device)
        else:
            q = axono.rope(q, pos, theta)
            k = axono.rope(k, pos, theta)
        attn = axono.scaled_dot_product_attention(q, k, v, True)
        attn = _reshape2(attn, seq, n_head * d_head, device)
        attn_out = axono.linear_nobias(attn, w[f"{p}.self_attn.o_proj.weight"])
        h = axono.add(h, attn_out)
        # --- mlp (SwiGLU) ---
        x = axono.rms_norm(h, w[f"{p}.post_attention_layernorm.weight"], eps)
        gate = axono.linear_nobias(x, w[f"{p}.mlp.gate_proj.weight"])
        up = axono.linear_nobias(x, w[f"{p}.mlp.up_proj.weight"])
        mlp = axono.linear_nobias(
            axono.mul(axono.silu(gate), up), w[f"{p}.mlp.down_proj.weight"]
        )
        h = axono.add(h, mlp)
        # deepstack: 前 len(deepstack) 层输出后, 在 image token 位置加视觉特征
        if deepstack is not None and li < len(deepstack):
            h_np = h.to("cpu").to_numpy() if device != "cpu" else h.to_numpy()
            ds_np = (
                deepstack[li].to("cpu").to_numpy()
                if not isinstance(deepstack[li], np.ndarray)
                else deepstack[li]
            )
            for j, ip in enumerate(image_positions):
                h_np[ip] += ds_np[j]
            h = to_t2(h_np, device)

    h = axono.rms_norm(h, w["model.language_model.norm.weight"], eps)
    # tied embedding: logits = h @ E^T — 用 linear (E 作为 weight)
    logits = axono.linear_nobias(h, w["model.language_model.embed_tokens.weight"])
    return logits.to("cpu").to_numpy()


def _ids(arr, device):
    arr = np.asarray(arr, dtype=np.int64)
    t = axono.Tensor.zeros(tuple(arr.shape), dtype=axono.DataType.INT64, device=device)
    t.copy_from_numpy(arr)
    return t


def _arange(seq, device):
    return _ids(np.arange(seq), device)


def to_t2(arr, device):
    t = axono.Tensor.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    return t.to(device) if device != "cpu" else t


def mrope_cos_sin(pos3: np.ndarray, theta: float, d_head: int):
    """M-RoPE cos/sin。pos3: (3, seq) int64。返回 (seq, d_head) cos/sin。

    HF 约定 (Qwen3VLTextRotaryEmbedding.recomposition_frequencies):
    每维 freqs = pos * inv_freq (half 个列); 重排 = 以 T 维为基础,
    H 覆盖 idx = 1..(section[1]*3) step 3, W 覆盖 idx = 2..(section[2]*3)
    step 3, 其余保持 T (交错 stride-3 布局, Qwen3-VL-2B
    mrope_section = [24, 20, 20] @ half=64); 最后 cat(f, f)。
    """
    inv_freq = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    freqs = (
        pos3[:, :, None].astype(np.float32) * inv_freq[None, None, :]
    )  # (3, seq, half)
    sec = [24, 20, 20]
    out = freqs[0].copy()  # (seq, half) — T 为基础
    idx_h = np.arange(1, sec[1] * 3, 3)
    idx_w = np.arange(2, sec[2] * 3, 3)
    out[:, idx_h] = freqs[1][:, idx_h]
    out[:, idx_w] = freqs[2][:, idx_w]
    angles = np.concatenate([out, out], axis=-1)
    return np.cos(angles), np.sin(angles)


def rope_cs(t3, cos, sin, device):
    """half-split rotate, cos/sin (seq, d_head) 按 head 广播。t3: (seq, heads, d_head)。"""
    arr = t3.to("cpu").to_numpy() if device != "cpu" else t3.to_numpy()
    seq, heads, d_head = arr.shape
    half = d_head // 2
    c = cos[:, None, :]
    s = sin[:, None, :]
    rot = np.concatenate([-arr[..., half:], arr[..., :half]], axis=-1)
    return to_t2(arr * c + rot * s, device)


def _reshape3(t, seq, heads, d_head, device):
    return t.reshape((seq, heads, d_head))


def _reshape2(t, seq, dim, device):
    return t.reshape((seq, dim))


def _head_norm(t3, weight, eps, seq, heads, d_head, device):
    """对 (seq, heads, d_head) 的最后一维做 rmsnorm (weight: d_head)"""
    arr = t3.reshape((-1, d_head)).to_numpy()
    wt = weight.to_numpy()
    normed = arr / np.sqrt((arr**2).mean(-1, keepdims=True) + eps) * wt
    return axono.Tensor.from_numpy(
        np.ascontiguousarray(normed.reshape(seq, heads, d_head))
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/qwen3vl-2b")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda":
        if not axono.cuda_available():
            print("无 CUDA, 回退 CPU")
            args.device = "cpu"
    axono.set_backend(args.device)

    print("加载权重 (bf16 -> fp32)...")
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    w = load_weights(os.path.join(args.model, "model.safetensors"), args.device)
    print(f"  {len(w)} 个张量")

    # 分词: 优先 tokenizers 库
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
        ids = tok.encode(PROMPT).ids
        print("tokenizers 编码:", ids)
    except ImportError:
        print("tokenizers 库不可用, 请 pip install tokenizers")
        return 1

    print(f"前向 ({args.device})...")
    logits = qwen3_text_forward(w, ids, args.device, cfg)
    last = logits[-1]
    top5 = np.argsort(-last)[:5]
    print("Top-5 token ids:", top5.tolist())

    # 与 torch 参考对比
    try:
        import torch
        from safetensors.torch import load_file

        tw = load_file(os.path.join(args.model, "model.safetensors"))
        tw = {k: v.float() for k, v in tw.items()}
        td = "cuda" if torch.cuda.is_available() else "cpu"
        tw = {k: v.to(td) for k, v in tw.items()}
        tid = torch.tensor([ids], device=td)
        with torch.no_grad():
            from transformers.models.qwen3_vl import (
                Qwen3VLForConditionalGeneration,
            )

            model = (
                Qwen3VLForConditionalGeneration.from_pretrained(
                    args.model, torch_dtype=torch.float32
                )
                .to(td)
                .eval()
            )
            ref = model(input_ids=tid).logits[0, -1].float().cpu().numpy()
        err = np.abs(last - ref).max()
        print(f"vs transformers 最后位置 logits max_abs_err = {err:.4e}")
        ok = err < 0.5
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    except ImportError as e:
        print(f"transformers 不可用 ({e}), 仅输出 top-5")
        return 0


if __name__ == "__main__":
    sys.exit(main())
