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
"""Torch 风格 Qwen3-VL-2B 模型 (纯 Axono nn.Module)。

支持 ``Qwen3VLForConditionalGeneration(config)`` 初始化 + ``load_state_dict``
加载 HF safetensors 权重, 与 Function 风格 (examples/qwen3vl_*_forward.py)
等价: 从像素/文本输入一路跑到 logits。

用法: PYTHONPATH=. python examples/qwen3vl_model_forward.py --device cpu|cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import axono
from axono.nn import Embedding, LayerNorm, Linear, Module, RMSNorm

MODEL = "/root/autodl-tmp/qwen3vl-2b"
IMAGE_TOKEN_ID = 151655


# ---------------------------------------------------------------------------
# M-RoPE 工具 (HF Qwen3VLTextRotaryEmbedding 交错 stride-3 重组约定)
# ---------------------------------------------------------------------------
def mrope_cos_sin(pos3: np.ndarray, theta: float, d_head: int):
    """pos3: (3, seq) -> cos/sin (seq, d_head)。T 基础, H/W 覆盖 idx%3==1/2。"""
    inv_freq = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    freqs = pos3[:, :, None].astype(np.float32) * inv_freq[None, None, :]
    sec = [24, 20, 20]
    out = freqs[0].copy()
    out[:, np.arange(1, sec[1] * 3, 3)] = freqs[1][:, np.arange(1, sec[1] * 3, 3)]
    out[:, np.arange(2, sec[2] * 3, 3)] = freqs[2][:, np.arange(2, sec[2] * 3, 3)]
    angles = np.concatenate([out, out], axis=-1)
    return np.cos(angles), np.sin(angles)


def build_mrope_pos3(ids, img_pos, want, llm_h, llm_w):
    """HF get_rope_index (单图): 文本段 3 维顺序计数; 图像段 raster。"""
    n = len(ids)
    img_start = int(img_pos[0])
    pos3 = np.zeros((3, n), dtype=np.int64)
    pos3[:, :img_start] = np.arange(img_start)[None, :]
    cur = img_start
    seg = np.arange(img_start, img_start + want)
    pos3[0, seg] = cur
    pos3[1, seg] = cur + np.arange(want) // llm_w
    pos3[2, seg] = cur + np.arange(want) % llm_w
    cur += max(llm_h, llm_w)
    pos3[:, img_start + want :] = (np.arange(n - (img_start + want)) + cur)[None, :]
    return pos3


def gelu_tanh(x, device):
    """0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3))) — numpy 复合 (tanh 近似式)。"""
    arr = x.to("cpu").to_numpy() if x.device != "cpu" else x.to_numpy()
    x3 = arr * arr * arr
    inner = np.sqrt(2 / np.pi) * (arr + 0.044715 * x3)
    out = 0.5 * arr * (1.0 + np.tanh(inner))
    t = axono.Tensor.from_numpy(np.ascontiguousarray(out.astype(np.float32)))
    return t.to(device) if device != "cpu" else t


def _head_norm(t3, weight, eps, seq, heads, d_head, device):
    """对 (seq, heads, d_head) 最后一维 RMSNorm。"""
    arr = t3.reshape((-1, d_head)).to_numpy()
    wt = weight.to_numpy()
    normed = arr / np.sqrt((arr**2).mean(-1, keepdims=True) + eps) * wt
    t = axono.Tensor.from_numpy(
        np.ascontiguousarray(normed.reshape(seq, heads, d_head))
    )
    return t.to(device) if device != "cpu" else t


def _to_t(arr, device):
    t = axono.Tensor.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    return t.to(device) if device != "cpu" else t


# ---------------------------------------------------------------------------
# 模型组件
# ---------------------------------------------------------------------------
class Qwen3VLVisionAttention(Module):
    def __init__(self, hidden: int, n_head: int, device=None):
        super().__init__()
        self.n_head = n_head
        self.d_head = hidden // n_head
        self.qkv = Linear(hidden, 3 * hidden, bias=True, device=device)
        self.proj = Linear(hidden, hidden, bias=True, device=device)

    def forward(self, x, cos_v, sin_v, n, device):
        qkv = self.qkv(x).to_numpy().reshape(n, 3, self.n_head, self.d_head)
        q = _to_t(qkv[:, 0], device)
        k = _to_t(qkv[:, 1], device)
        v = _to_t(qkv[:, 2], device)
        for name, t in (("q", q), ("k", k)):
            arr = t.to_numpy().reshape(n, self.n_head, self.d_head)
            half = self.d_head // 2
            c = cos_v[:, None, :]
            s = sin_v[:, None, :]
            rot = np.concatenate([-arr[..., half:], arr[..., :half]], axis=-1)
            out = arr * c + rot * s
            tt = _to_t(out, device)
            if name == "q":
                q = tt
            else:
                k = tt
        attn = axono.scaled_dot_product_attention(q, k, v, False)
        return self.proj(attn.reshape((n, -1)))


class Qwen3VLVisionMLP(Module):
    def __init__(self, hidden: int, inter: int, device=None):
        super().__init__()
        self.linear_fc1 = Linear(hidden, inter, bias=True, device=device)
        self.linear_fc2 = Linear(inter, hidden, bias=True, device=device)

    def forward(self, x, device):
        return self.linear_fc2.forward(gelu_tanh(self.linear_fc1.forward(x), device))


class Qwen3VLVisionBlock(Module):
    def __init__(self, hidden: int, n_head: int, inter: int, device=None):
        super().__init__()
        self.norm1 = LayerNorm(hidden, eps=1e-6, device=device)
        self.norm2 = LayerNorm(hidden, eps=1e-6, device=device)
        self.attn = Qwen3VLVisionAttention(hidden, n_head, device=device)
        self.mlp = Qwen3VLVisionMLP(hidden, inter, device=device)

    def forward(self, x, cos_v, sin_v, n, device):
        xn1 = self.norm1(x)
        x = axono.add(x, self.attn(xn1, cos_v, sin_v, n, device))
        xn2 = self.norm2(x)
        x = axono.add(x, self.mlp(xn2, device))
        return x


class Qwen3VLMerger(Module):
    def __init__(
        self,
        hidden: int,
        merge: int,
        out_hidden: int,
        device=None,
        postshuffle_norm: bool = False,
    ):
        super().__init__()
        hs = hidden * merge * merge
        self.postshuffle_norm = postshuffle_norm
        # HF: 主 merger norm 在 view 之前 (hidden); deepstack (use_postshuffle_norm)
        # norm 在 view 之后 (hs)。
        self.norm = LayerNorm(
            hs if postshuffle_norm else hidden, eps=1e-6, device=device
        )
        self.linear_fc1 = Linear(hs, hs, bias=True, device=device)
        self.linear_fc2 = Linear(hs, out_hidden, bias=True, device=device)

    def forward(self, x, postshuffle: bool, hs: int, device):
        if postshuffle:
            arr = x.to("cpu").to_numpy() if x.device != "cpu" else x.to_numpy()
            x = _to_t(arr.reshape(-1, hs), device)
            x = self.norm(x)
        else:
            x = self.norm(x)
            arr = x.to("cpu").to_numpy() if x.device != "cpu" else x.to_numpy()
            x = _to_t(arr.reshape(-1, hs), device)
        return self.linear_fc2(axono.gelu(self.linear_fc1(x)))


class Qwen3VLPatchEmbed(Module):
    """conv3d(t=2,p=16,s=16) 等价: 权重展平 + linear。"""

    def __init__(
        self, hidden: int, in_ch: int = 3, t: int = 2, patch: int = 16, device=None
    ):
        super().__init__()
        self.in_feat = in_ch * t * patch * patch
        self.proj = Linear(self.in_feat, hidden, bias=True, device=device)

    def forward(self, pixels: np.ndarray, device):
        n = pixels.shape[0]
        x = _to_t(pixels.reshape(n, -1), device)
        return self.proj(x)


class Qwen3VLVisionModel(Module):
    def __init__(self, cfg: dict, device=None):
        super().__init__()
        v = cfg["vision_config"]
        hidden = v["hidden_size"]
        self.cfg_v = v
        self.patch_size = v["patch_size"]
        self.temporal_patch_size = v["temporal_patch_size"]
        self.spatial_merge_size = v["spatial_merge_size"]
        self.n_head = v["num_heads"]
        self.depth = v["depth"]
        self.out_hidden = cfg["text_config"]["hidden_size"]
        self.eps = 1e-6
        self.pos_side = int(v["num_position_embeddings"] ** 0.5)

        self.patch_embed = Qwen3VLPatchEmbed(
            hidden,
            t=self.temporal_patch_size,
            patch=self.patch_size,
            device=device,
        )
        self.pos_embed = Embedding(v["num_position_embeddings"], hidden, device=device)
        inter = v.get("intermediate_size", hidden * 4)
        self.blocks = [
            Qwen3VLVisionBlock(hidden, self.n_head, inter, device=device)
            for _ in range(self.depth)
        ]
        self.merger = Qwen3VLMerger(
            hidden, self.spatial_merge_size, self.out_hidden, device=device
        )
        self.deepstack_merger_list = [
            Qwen3VLMerger(
                hidden,
                self.spatial_merge_size,
                self.out_hidden,
                device=device,
                postshuffle_norm=True,
            )
            for _ in v["deepstack_visual_indexes"]
        ]

    def forward(
        self, pixels: np.ndarray, grid_t: int, grid_h: int, grid_w: int, device: str
    ):
        v = self.cfg_v
        merge = self.spatial_merge_size
        n = grid_t * grid_h * grid_w
        hidden = self.pos_embed.embedding_dim

        x = self.patch_embed.forward(pixels, device)

        # pos_embed 双线性插值 (align_corners=True, merge-block token 序)
        rb, cb = grid_h // merge, grid_w // merge
        r_idx = np.arange(grid_h).reshape(rb, merge)
        c_idx = np.arange(grid_w).reshape(cb, merge)
        seq = [
            (r_idx[a, ir], c_idx[b, ic])
            for a in range(rb)
            for b in range(cb)
            for ir in range(merge)
            for ic in range(merge)
        ]
        seq = np.array(seq)
        rows_i, cols_i = seq[:, 0], seq[:, 1]

        pe = (
            self.pos_embed.weight.to("cpu")
            .to_numpy()
            .reshape(self.pos_side, self.pos_side, hidden)
        )
        ys = np.linspace(0, self.pos_side - 1, grid_h)
        xs = np.linspace(0, self.pos_side - 1, grid_w)
        pos = np.empty((n, hidden), dtype=np.float32)
        for i, (r, c) in enumerate(zip(rows_i, cols_i)):
            ry, rx = ys[r], xs[c]
            iy, ix = min(int(ry), self.pos_side - 2), min(int(rx), self.pos_side - 2)
            fy, fx = ry - iy, rx - ix
            pos[i] = (
                pe[iy, ix] * (1 - fy) * (1 - fx)
                + pe[iy, ix + 1] * (1 - fy) * fx
                + pe[iy + 1, ix] * fy * (1 - fx)
                + pe[iy + 1, ix + 1] * fy * fx
            )
        x = axono.add(x, _to_t(pos, device))

        # 2D axial rope
        rope_theta = v.get("rope_parameters", {}).get("rope_theta", 10000.0)
        d_head = hidden // self.n_head
        inv_freq = rope_theta ** (
            -np.arange(0, d_head // 2, 2, dtype=np.float32) / (d_head // 2)
        )
        fh = rows_i.astype(np.float32)[:, None] * inv_freq[None, :]
        fw = cols_i.astype(np.float32)[:, None] * inv_freq[None, :]
        fhw = np.concatenate([fh, fw], axis=-1)
        angles = np.concatenate([fhw, fhw], axis=-1)
        cos_v = np.cos(angles).astype(np.float32)
        sin_v = np.sin(angles).astype(np.float32)

        # blocks
        deepstack = []
        ds_idx = set(v["deepstack_visual_indexes"])
        for li, blk in enumerate(self.blocks):
            x = blk.forward(x, cos_v, sin_v, n, device)
            if li in ds_idx:
                deepstack.append(x)

        hs = hidden * merge * merge
        merged = self.merger.forward(x, False, hs, device)
        ds_out = [
            m.forward(d, True, hs, device)
            for m, d in zip(self.deepstack_merger_list, deepstack)
        ]
        return merged, ds_out


class Qwen3VLTextAttention(Module):
    def __init__(self, hidden: int, n_head: int, n_kv: int, d_head: int, device=None):
        super().__init__()
        self.n_head = n_head
        self.n_kv = n_kv
        self.d_head = d_head
        self.q_proj = Linear(hidden, n_head * d_head, bias=False, device=device)
        self.k_proj = Linear(hidden, n_kv * d_head, bias=False, device=device)
        self.v_proj = Linear(hidden, n_kv * d_head, bias=False, device=device)
        self.o_proj = Linear(n_head * d_head, hidden, bias=False, device=device)
        self.q_norm = RMSNorm(d_head, eps=1e-6, device=device)
        self.k_norm = RMSNorm(d_head, eps=1e-6, device=device)

    def forward(self, x, seq, cos_m, sin_m, device):
        q = self.q_proj(x).reshape((seq, self.n_head, self.d_head))
        k = self.k_proj(x).reshape((seq, self.n_kv, self.d_head))
        v = self.v_proj(x).reshape((seq, self.n_kv, self.d_head))
        q = _head_norm(
            q,
            self.q_norm.weight,
            self.q_norm.eps,
            seq,
            self.n_head,
            self.d_head,
            device,
        )
        k = _head_norm(
            k, self.k_norm.weight, self.k_norm.eps, seq, self.n_kv, self.d_head, device
        )
        if cos_m is not None:
            q = self._rope(q, cos_m, sin_m, device)
            k = self._rope(k, cos_m, sin_m, device)
        else:
            pos = axono.Tensor.zeros((seq,), dtype=axono.DataType.INT64)
            pos.copy_from_numpy(np.arange(seq, dtype=np.int64))
            q = axono.rope(q, pos, 10000.0)
            k = axono.rope(k, pos, 10000.0)
        attn = axono.scaled_dot_product_attention(q, k, v, True)
        return self.o_proj(attn.reshape((seq, -1)))

    @staticmethod
    def _rope(t3, cos, sin, device):
        arr = t3.to("cpu").to_numpy() if t3.device != "cpu" else t3.to_numpy()
        half = arr.shape[-1] // 2
        c = cos[:, None, :]
        s = sin[:, None, :]
        rot = np.concatenate([-arr[..., half:], arr[..., :half]], axis=-1)
        return _to_t(arr * c + rot * s, device)


class Qwen3VLTextMLP(Module):
    def __init__(self, hidden: int, inter: int, device=None):
        super().__init__()
        self.gate_proj = Linear(hidden, inter, bias=False, device=device)
        self.up_proj = Linear(hidden, inter, bias=False, device=device)
        self.down_proj = Linear(inter, hidden, bias=False, device=device)

    def forward(self, x):
        return self.down_proj(axono.mul(axono.silu(self.gate_proj(x)), self.up_proj(x)))


class Qwen3VLDecoderLayer(Module):
    def __init__(
        self,
        hidden: int,
        n_head: int,
        n_kv: int,
        d_head: int,
        inter: int,
        eps: float,
        device=None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps=eps, device=device)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps, device=device)
        self.self_attn = Qwen3VLTextAttention(
            hidden, n_head, n_kv, d_head, device=device
        )
        self.mlp = Qwen3VLTextMLP(hidden, inter, device=device)

    def forward(self, h, seq, cos_m, sin_m, device):
        xn = self.input_layernorm(h)
        h = axono.add(h, self.self_attn.forward(xn, seq, cos_m, sin_m, device))
        xn2 = self.post_attention_layernorm(h)
        h = axono.add(h, self.mlp.forward(xn2))
        return h


class Qwen3VLTextModel(Module):
    def __init__(self, cfg: dict, device=None):
        super().__init__()
        t = cfg["text_config"]
        self.cfg_t = t
        self.hidden = t["hidden_size"]
        self.n_head = t["num_attention_heads"]
        self.n_kv = t["num_key_value_heads"]
        self.d_head = t["head_dim"]
        self.layers_n = t["num_hidden_layers"]
        self.eps = t["rms_norm_eps"]
        self.theta = float(t["rope_theta"])
        self.inter = t["intermediate_size"]
        self.vocab = t["vocab_size"]

        self.embed_tokens = Embedding(self.vocab, self.hidden, device=device)
        self.layers = [
            Qwen3VLDecoderLayer(
                self.hidden,
                self.n_head,
                self.n_kv,
                self.d_head,
                self.inter,
                self.eps,
                device=device,
            )
            for _ in range(self.layers_n)
        ]
        self.norm = RMSNorm(self.hidden, eps=self.eps, device=device)

    def forward(
        self,
        ids,
        hidden=None,
        deepstack=None,
        image_positions=None,
        mrope_pos3=None,
        device="cpu",
    ):
        seq = len(ids)
        if hidden is not None:
            h = _to_t(hidden, device)
        else:
            ids_t = axono.Tensor.zeros(
                (1, seq), dtype=axono.DataType.INT64, device=device
            )
            ids_t.copy_from_numpy(np.asarray([ids], dtype=np.int64))
            h = self.embed_tokens.forward(ids_t).reshape((seq, self.hidden))

        cos_m = sin_m = None
        if mrope_pos3 is not None:
            cos_m, sin_m = mrope_cos_sin(mrope_pos3, self.theta, self.d_head)

        ds_n = len(deepstack) if deepstack is not None else 0
        for li, layer in enumerate(self.layers):
            h = layer.forward(h, seq, cos_m, sin_m, device)
            if li < ds_n:
                h_np = h.to("cpu").to_numpy() if device != "cpu" else h.to_numpy()
                ds_np = (
                    deepstack[li].to("cpu").to_numpy()
                    if not isinstance(deepstack[li], np.ndarray)
                    else deepstack[li]
                )
                for j, ip in enumerate(image_positions):
                    h_np[ip] += ds_np[j]
                h = _to_t(h_np, device)

        h = self.norm.forward(h)
        logits = axono.linear_nobias(h, self.embed_tokens.weight)
        return logits.to("cpu").to_numpy()


class Qwen3VLForConditionalGeneration(Module):
    """Torch 风格封装: 视觉塔 + 文本塔, 从 pixels+ids 到 logits。"""

    def __init__(self, config_path: str, device: str | None = None):
        super().__init__()
        device = device or ("cuda" if axono.cuda_available() else "cpu")
        axono.set_backend(device)
        self.device = device
        with open(config_path) as f:
            self.config = json.load(f)
        self.model = _Inner(self.config, device)

    def get_input_embeddings(self):
        return self.model.text.embed_tokens

    def forward(
        self,
        pixels: np.ndarray,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        ids,
        deepstack=True,
    ):
        """pixels: (n, 3, t, patch, patch); ids: 含 IMAGE_TOKEN_ID 的文本序列。"""
        merged, ds = self.model.visual.forward(pixels, 1, grid_h, grid_w, self.device)
        want = (grid_h // self.model.visual.spatial_merge_size) * (
            grid_w // self.model.visual.spatial_merge_size
        )
        img_pos = np.arange(
            ids.index(IMAGE_TOKEN_ID), ids.index(IMAGE_TOKEN_ID) + want, dtype=np.int64
        )
        merge = self.model.visual.spatial_merge_size
        pos3 = build_mrope_pos3(ids, img_pos, want, grid_h // merge, grid_w // merge)

        # embedding + image token 替换
        emb = self.model.text.embed_tokens
        ids_t = axono.Tensor.zeros(
            (1, len(ids)), dtype=axono.DataType.INT64, device=self.device
        )
        ids_t.copy_from_numpy(np.asarray([ids], dtype=np.int64))
        h = emb.forward(ids_t).reshape((len(ids), -1)).to_numpy()
        m_np = (
            merged.to("cpu").to_numpy()
            if merged.device != "cpu"
            else (merged.to_numpy())
        )
        for j, ip in enumerate(img_pos):
            h[ip] = m_np[j]

        logits = self.model.text.forward(
            ids,
            hidden=h,
            deepstack=ds if deepstack else None,
            image_positions=img_pos,
            mrope_pos3=pos3,
            device=self.device,
        )
        return logits


class _Inner(Module):
    """visual / text 两个子塔容器。"""

    def __init__(self, cfg: dict, device=None):
        super().__init__()
        self.cfg_v_spatial_merge = cfg["vision_config"]["spatial_merge_size"]
        self.visual = Qwen3VLVisionModel(cfg, device=device)
        self.text = Qwen3VLTextModel(cfg, device=device)


def load_hf_state_dict(model_dir: str) -> dict:
    """HF safetensors -> {扁平名: fp32 ndarray}。"""
    from safetensors.torch import load_file

    raw = load_file(os.path.join(model_dir, "model.safetensors"))
    return {k: v.float().cpu().numpy() for k, v in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not axono.cuda_available():
        print("无 CUDA, 回退 CPU")
        device = "cpu"

    cfg_path = os.path.join(args.model, "config.json")
    print(f"初始化 Qwen3VLForConditionalGeneration ({device}) ...")
    model = Qwen3VLForConditionalGeneration(cfg_path, device=device)

    print("加载 HF safetensors 权重 ...")
    sd = load_hf_state_dict(args.model)
    # conv3d 权重 (out, c, t, ph, pw) -> 展平为 (out, c*t*ph*pw) 供 Linear 加载
    sd = {
        k: (v.reshape(v.shape[0], -1) if k.endswith("patch_embed.proj.weight") else v)
        for k, v in sd.items()
    }
    model.model.visual.load_state_dict(
        {
            k[len("model.visual.") :]: v
            for k, v in sd.items()
            if k.startswith("model.visual.")
        }
    )
    model.model.text.load_state_dict(
        {
            k[len("model.language_model.") :]: v
            for k, v in sd.items()
            if k.startswith("model.language_model.")
        }
    )
    print(f"  参数量: {len(model.parameters())}")

    # 输入: 256x256 随机图 + 多模态 prompt
    v = model.config["vision_config"]
    patch = v["patch_size"]
    grid_h = grid_w = 256 // patch
    n = grid_h * grid_w
    want = n // (v["spatial_merge_size"] ** 2)
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal((n, 3, v["temporal_patch_size"], patch, patch)).astype(
        np.float32
    )

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    ids = tok.encode("<|im_start|>user\n").ids
    ids = ids + [IMAGE_TOKEN_ID] * want
    ids = ids + tok.encode("描述这张图片<|im_end|>\n<|im_start|>assistant\n").ids
    print(f"input_ids: {len(ids)} tokens, image tokens: {want}")

    print("端到端前向 (Module 版) ...")
    logits = model.forward(pixels, 1, grid_h, grid_w, ids)
    last = logits[-1] if logits.ndim == 2 else logits[0, -1]
    print("Top-5:", np.argsort(-last)[:5].tolist())

    if args.no_ref:
        return 0

    import torch
    from transformers.models.qwen3_vl import (
        Qwen3VLForConditionalGeneration as _HFModel,
    )

    ref = _HFModel.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        r = ref(
            input_ids=torch.tensor([ids]),
            pixel_values=torch.from_numpy(
                pixels.reshape(n, 3 * v["temporal_patch_size"], patch, patch)
            ),
            image_grid_thw=torch.tensor([[1, grid_h, grid_w]]),
            mm_token_type_ids=torch.tensor(
                [[1 if t == IMAGE_TOKEN_ID else 0 for t in ids]]
            ),
        )
    ref_lg = r.logits[0, -1].float().numpy()
    err = np.abs(last - ref_lg).max()
    rel = err / np.abs(ref_lg).max()
    print(f"端到端 logits max_abs_err = {err:.4e} (rel {rel:.2e})")
    print("HF Top-5:", np.argsort(-ref_lg)[:5].tolist())
    ok = err < 5e-2 and np.array_equal(np.argsort(-last)[:1], np.argsort(-ref_lg)[:1])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
