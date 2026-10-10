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
"""Torch 风格 Qwen3-VL-2B (纯 Axono nn.Module, 高性能版)。

- 全程 Tensor API, 前向无 numpy 往返 (cos/sin 等常量只建一次并缓存);
- gelu_tanh / rope_with_cos_sin / rope_thd 走 C++/CUDA 单算子;
- M-RoPE 用 rope_thd (交错 stride-3 频率重组, HF 等价);
- SDPA CUDA kernel block 内并行 (warp 分片点积 + shuffle 归约);
- Linear 单算子融合 (CPU cblas / CUDA cuBLASLt)。
"""

from __future__ import annotations

import json
import os

import numpy as np

import axono
from axono import Tensor, nn

IMAGE_TOKEN_ID = 151655


# ---------------------------------------------------------------------------
# 位置/频率工具 (仅构建期少量 numpy, 结果缓存为 Tensor)
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


def build_mrope_pos3(ids, img_start: int, want: int, llm_h: int, llm_w: int):
    """HF get_rope_index (单图): 文本段 3 维顺序计数; 图像段 raster。"""
    n = len(ids)
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


def _t(arr: np.ndarray, device: str) -> axono.Tensor:
    """numpy -> Tensor (仅入口/常量构建用)。"""
    t = axono.Tensor.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    return t.to(device) if device != "cpu" else t


def _copy(t: axono.Tensor) -> axono.Tensor:
    """Tensor 浅复制 (reshape 是元数据原地操作, 需要 reshape 前复制)。"""
    out = axono.Tensor(t.dtype, list(t.shape), t.device)
    out.copy_from(t)
    return out


# ---------------------------------------------------------------------------
# 视觉塔
# ---------------------------------------------------------------------------
class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, hidden: int, n_head: int, device=None):
        super().__init__()
        self.n_head = n_head
        self.d_head = hidden // n_head
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True, device=device)
        self.proj = nn.Linear(hidden, hidden, bias=True, device=device)

    def forward(self, x, cos_v, sin_v, n, device):
        h = self.n_head
        d = self.d_head
        qkv = self.qkv(x)  # (n, 3*h*d)
        # q/k/v 沿 last 维切 (SliceLastDimKernel 快路径), 免去整块 copy:
        # q = 列 [0, hd), k = [hd, 2hd), v = [2hd, 3hd) — 与 HF 布局一致
        hd = h * d
        q = axono.slice_(qkv, -1, 0, hd).reshape((n, h, d))
        k = axono.slice_(qkv, -1, hd, hd).reshape((n, h, d))
        v = axono.slice_(qkv, -1, 2 * hd, hd).reshape((n, h, d))
        q = axono.rope_with_cos_sin(q, cos_v, sin_v)
        k = axono.rope_with_cos_sin(k, cos_v, sin_v)
        attn = axono.scaled_dot_product_attention(q, k, v, False)
        return self.proj(attn.reshape((n, h * d)))


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, device=None):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden, inter, bias=True, device=device)
        self.linear_fc2 = nn.Linear(inter, hidden, bias=True, device=device)

    def forward(self, x):
        return self.linear_fc2(axono.gelu_tanh(self.linear_fc1(x)))


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, hidden: int, n_head: int, inter: int, device=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, eps=1e-6, device=device)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6, device=device)
        self.attn = Qwen3VLVisionAttention(hidden, n_head, device=device)
        self.mlp = Qwen3VLVisionMLP(hidden, inter, device=device)

    def forward(self, x, cos_v, sin_v, n, device):
        xn1 = self.norm1(x)
        x = axono.add(x, self.attn(xn1, cos_v, sin_v, n, device))
        xn2 = self.norm2(x)
        x = axono.add(x, self.mlp(xn2))
        return x


class Qwen3VLMerger(nn.Module):
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
        self.hs = hs
        self.postshuffle_norm = postshuffle_norm
        # HF: 主 merger norm 在 view 之前 (hidden); deepstack (postshuffle) 在 view 之后。
        self.norm = nn.LayerNorm(
            hs if postshuffle_norm else hidden, eps=1e-6, device=device
        )
        self.linear_fc1 = nn.Linear(hs, hs, bias=True, device=device)
        self.linear_fc2 = nn.Linear(hs, out_hidden, bias=True, device=device)

    def forward(self, x, postshuffle: bool):
        if postshuffle:
            # deepstack: norm 在 view 之后 (hs = hidden*merge^2)
            x = _copy(x).reshape((-1, self.hs))
            x = self.norm(x)
        else:
            # 主 merger: norm 在 view 之前 (hidden)
            x = self.norm(x)
            x = _copy(x).reshape((-1, self.hs))
        return self.linear_fc2(axono.gelu(self.linear_fc1(x)))


class Qwen3VLPatchEmbed(nn.Module):
    """conv3d(t=2,p=16,s=16) 等价: 权重展平 + linear。"""

    def __init__(
        self, hidden: int, in_ch: int = 3, t: int = 2, patch: int = 16, device=None
    ):
        super().__init__()
        self.in_feat = in_ch * t * patch * patch
        self.proj = nn.Linear(self.in_feat, hidden, bias=True, device=device)

    def forward(self, pixels: np.ndarray, device):
        n = pixels.shape[0]
        return self.proj(_t(pixels.reshape(n, -1), device))


class Qwen3VLVisionModel(nn.Module):
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
        self.pos_side = int(v["num_position_embeddings"] ** 0.5)
        self.d_head = hidden // self.n_head

        self.patch_embed = Qwen3VLPatchEmbed(
            hidden,
            t=self.temporal_patch_size,
            patch=self.patch_size,
            device=device,
        )
        self.pos_embed = nn.Embedding(
            v["num_position_embeddings"], hidden, device=device
        )
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
        self._pos_cache: dict = {}

    # -- 常量缓存: pos 插值表与 2D rope cos/sin (同 grid 只算一次) --
    def _grid_consts(self, grid_t, grid_h, grid_w, device):
        key = (grid_t, grid_h, grid_w, device)
        if key in self._pos_cache:
            return self._pos_cache[key]
        v = self.cfg_v
        merge = self.spatial_merge_size
        n = grid_t * grid_h * grid_w
        hidden = self.pos_embed.embedding_dim

        rb, cb = grid_h // merge, grid_w // merge
        r_idx = np.arange(grid_h).reshape(rb, merge)
        c_idx = np.arange(grid_w).reshape(cb, merge)
        seq = np.array(
            [
                (r_idx[a, ir], c_idx[b, ic])
                for a in range(rb)
                for b in range(cb)
                for ir in range(merge)
                for ic in range(merge)
            ]
        )
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

        rope_theta = v.get("rope_parameters", {}).get("rope_theta", 10000.0)
        inv_freq = rope_theta ** (
            -np.arange(0, self.d_head // 2, 2, dtype=np.float32) / (self.d_head // 2)
        )
        fh = rows_i.astype(np.float32)[:, None] * inv_freq[None, :]
        fw = cols_i.astype(np.float32)[:, None] * inv_freq[None, :]
        fhw = np.concatenate([fh, fw], axis=-1)
        angles = np.concatenate([fhw, fhw], axis=-1)
        cos_v = np.cos(angles).astype(np.float32)
        sin_v = np.sin(angles).astype(np.float32)

        out = (
            _t(pos, device),
            _t(cos_v, device),
            _t(sin_v, device),
        )
        self._pos_cache[key] = out
        return out

    def forward(
        self, pixels: np.ndarray, grid_t: int, grid_h: int, grid_w: int, device: str
    ):
        v = self.cfg_v
        n = grid_t * grid_h * grid_w

        x = self.patch_embed(pixels, device)
        pos, cos_v, sin_v = self._grid_consts(grid_t, grid_h, grid_w, device)
        x = axono.add(x, pos)

        deepstack = []
        ds_idx = set(v["deepstack_visual_indexes"])
        for li, blk in enumerate(self.blocks):
            x = blk(x, cos_v, sin_v, n, device)
            if li in ds_idx:
                deepstack.append(x)

        merged = self.merger(x, False)
        ds_out = [m(d, True) for m, d in zip(self.deepstack_merger_list, deepstack)]
        return merged, ds_out


# ---------------------------------------------------------------------------
# 文本塔
# ---------------------------------------------------------------------------
class Qwen3VLTextAttention(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_head: int,
        n_kv: int,
        d_head: int,
        use_gqa: bool = True,
        device=None,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_kv = n_kv
        self.d_head = d_head
        self.use_gqa = use_gqa
        self.q_proj = nn.Linear(hidden, n_head * d_head, bias=False, device=device)
        self.k_proj = nn.Linear(hidden, n_kv * d_head, bias=False, device=device)
        self.v_proj = nn.Linear(hidden, n_kv * d_head, bias=False, device=device)
        self.o_proj = nn.Linear(n_head * d_head, hidden, bias=False, device=device)
        self.q_norm = nn.RMSNorm(d_head, eps=1e-6, device=device)
        self.k_norm = nn.RMSNorm(d_head, eps=1e-6, device=device)

    def forward(
        self,
        x,
        seq,
        inv_freq_t,
        mrope_section,
        pos_t,
        device,
        cache=None,
        layer_idx=0,
    ):
        if pos_t is None:
            # 纯文本: 顺序位置 rope
            pos = axono.Tensor.zeros((seq,), dtype=axono.DataType.INT64, device=device)
            pos.copy_from_numpy(np.arange(seq, dtype=np.int64))
            q = self.q_proj(x).reshape((seq, self.n_head, self.d_head))
            k = self.k_proj(x).reshape((seq, self.n_kv, self.d_head))
            v = self.v_proj(x).reshape((seq, self.n_kv, self.d_head))
            q = axono.rms_norm(q, self.q_norm.weight, self.q_norm.eps)
            k = axono.rms_norm(k, self.k_norm.weight, self.k_norm.eps)
            q = axono.rope(q, pos, 10000.0)
            k = axono.rope(k, pos, 10000.0)
        else:
            q = self.q_proj(x).reshape((seq, self.n_head, self.d_head))
            k = self.k_proj(x).reshape((seq, self.n_kv, self.d_head))
            v = self.v_proj(x).reshape((seq, self.n_kv, self.d_head))
            q = axono.rms_norm(q, self.q_norm.weight, self.q_norm.eps)
            k = axono.rms_norm(k, self.k_norm.weight, self.k_norm.eps)
            q = axono.rope_thd(q, pos_t, inv_freq_t, mrope_section)
            k = axono.rope_thd(k, pos_t, inv_freq_t, mrope_section)
        if cache is not None:
            # KV cache 路径: k/v 追加进 cache, 注意力对「全历史」做 (非因果,
            # decode 时 q_len=1 只看过去; prefill 时 q_len=seq 自身即因果)。
            cache.update(layer_idx, k, v)
            k, v = cache.get(layer_idx)
            is_causal = seq == int(k.shape[0])
        else:
            is_causal = True
        if self.use_gqa and seq == 1 and cache is not None and int(k.shape[0]) > 1:
            # decode (q_len=1): GQA split-K 优化 kernel (flash-decoding)
            attn = axono.gqa_decode_attention(q, k, v)
        else:
            attn = axono.scaled_dot_product_attention(q, k, v, is_causal)
        return self.o_proj(attn.reshape((seq, self.n_head * self.d_head)))


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, device=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False, device=device)
        self.up_proj = nn.Linear(hidden, inter, bias=False, device=device)
        self.down_proj = nn.Linear(inter, hidden, bias=False, device=device)

    def forward(self, x):
        return self.down_proj(axono.fused.silu_mul(self.gate_proj(x), self.up_proj(x)))


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_head: int,
        n_kv: int,
        d_head: int,
        inter: int,
        eps: float,
        use_gqa: bool = True,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps, device=device)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps, device=device)
        self.self_attn = Qwen3VLTextAttention(
            hidden, n_head, n_kv, d_head, use_gqa=use_gqa, device=device
        )
        self.mlp = Qwen3VLTextMLP(hidden, inter, device=device)

    def forward(
        self,
        h,
        seq,
        inv_freq_t,
        mrope_section,
        pos_t,
        device,
        cache=None,
        layer_idx=0,
    ):
        attn_out = self.self_attn(
            self.input_layernorm(h),
            seq,
            inv_freq_t,
            mrope_section,
            pos_t,
            device,
            cache=cache,
            layer_idx=layer_idx,
        )
        # 融合: h = h + attn_out; xn2 = rmsnorm(h) (单 kernel, 省一轮读写)
        h, xn2 = axono.fused.add_rms_norm(
            attn_out, h, self.post_attention_layernorm.weight, self.eps
        )
        mlp_out = self.mlp(xn2)
        # 末层残差无需再 norm — 用普通加法
        h = axono.add(h, mlp_out)
        return h


class Qwen3VLTextModel(nn.Module):
    def __init__(self, cfg: dict, use_gqa: bool = True, device=None):
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
        self.mrope_section = list(
            t.get("rope_scaling", {}).get("mrope_section", [24, 20, 20])
        )

        self.embed_tokens = nn.Embedding(self.vocab, self.hidden, device=device)
        self.layers = [
            Qwen3VLDecoderLayer(
                self.hidden,
                self.n_head,
                self.n_kv,
                self.d_head,
                self.inter,
                self.eps,
                use_gqa=use_gqa,
                device=device,
            )
            for _ in range(self.layers_n)
        ]
        self.norm = nn.RMSNorm(self.hidden, eps=self.eps, device=device)
        self._inv_freq_t: dict = {}

    def _inv_freq(self, device):
        if device not in self._inv_freq_t:
            inv = 1.0 / (
                self.theta
                ** (np.arange(0, self.d_head, 2, dtype=np.float32) / self.d_head)
            )
            self._inv_freq_t[device] = _t(inv, device)
        return self._inv_freq_t[device]

    def final_norm_logits(self, h):
        """最终 norm + lm_head (weight tying)。"""
        h = self.norm(h)
        return axono.linear_nobias(h, self.embed_tokens.weight)

    def forward_layers(self, h, seq, inv_freq_t, section, pos_t, device, cache=None):
        """逐层前向 (m-rope 版, 供外部注入 deepstack)。"""
        for li, layer in enumerate(self.layers):
            h = layer.forward(
                h,
                seq,
                inv_freq_t,
                section,
                pos_t,
                device,
                cache=cache,
                layer_idx=li,
            )
        return h

    def forward_layers_with_deepstack(
        self,
        h,
        seq,
        inv_freq_t,
        section,
        pos_t,
        deepstack,
        img_start,
        want,
        device,
        cache=None,
    ):
        """逐层前向; 每层完整 (attn+mlp) 之后注入 deepstack (HF 语义)。"""
        pre = img_start
        post = seq - (img_start + want)
        ds_n = len(deepstack)
        for li, layer in enumerate(self.layers):
            h = layer.forward(
                h,
                seq,
                inv_freq_t,
                section,
                pos_t,
                device,
                cache=cache,
                layer_idx=li,
            )
            if li < ds_n:
                segs = []
                if pre:
                    segs.append(axono.slice_(h, 0, 0, pre))
                seg = axono.add(axono.slice_(h, 0, pre, want), deepstack[li])
                segs.append(seg)
                if post:
                    segs.append(axono.slice_(h, 0, pre + want, post))
                if len(segs) == 1:
                    h = segs[0]
                else:
                    h = segs[0]
                    for s in segs[1:]:
                        h = axono.concat(h, s, 0)
        return h


class Qwen3VLForConditionalGeneration(nn.Module):
    """Torch 风格封装: 视觉塔 + 文本塔, 从 pixels+ids 到 logits。"""

    def __init__(
        self, config_path: str, use_gqa: bool = True, device: str | None = None
    ):
        super().__init__()
        device = device or ("cuda" if axono.cuda_available() else "cpu")
        axono.set_backend(device)
        self.device = device
        with open(config_path) as f:
            self.config = json.load(f)
        self.visual = Qwen3VLVisionModel(self.config, device=device)
        self.text = Qwen3VLTextModel(self.config, use_gqa=use_gqa, device=device)

    def enable_fp16_linear(self) -> None:
        """Linear 权重转 fp16 常驻 (tensor-core 混合推理, 显存减半)。

        加载权重之后、推理之前调用; 输入输出 dtype 不变 (fp32)。
        """
        self.cast_linear_fp16()

    def load_hf_weights(self, model_dir: str) -> None:
        sd = load_hf_state_dict(model_dir)
        sd = {
            k: (
                v.reshape(v.shape[0], -1)
                if k.endswith("patch_embed.proj.weight")
                else v
            )
            for k, v in sd.items()
        }
        self.visual.load_state_dict(
            {
                k[len("model.visual.") :]: v
                for k, v in sd.items()
                if k.startswith("model.visual.")
            }
        )
        self.text.load_state_dict(
            {
                k[len("model.language_model.") :]: v
                for k, v in sd.items()
                if k.startswith("model.language_model.")
            }
        )

    def load_axm_weights(self, axm_path: str) -> None:
        """加载 .axm 原生权重 (memmap 零拷贝 + 设备直拷, 显著快于 HF 路径)。

        fp16 存储的张量在此转回 fp32 (ndarray astype 后仍走批量拷贝)。
        """
        from axono.format import AxmReader

        reader = AxmReader(axm_path)
        prefix_map = {"model.visual.": self.visual, "model.language_model.": self.text}
        bufs = {p: {} for p in prefix_map}
        keepalive = []
        for name in reader.names():
            for prefix, module in prefix_map.items():
                if name.startswith(prefix):
                    arr = reader.get(name)
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    # 零拷贝借用 memmap 内存: load_state_dict 里
                    # Tensor.to(cuda) 直接从文件映射 DMA, 省去 host 中转。
                    t = Tensor.from_numpy_ref(np.ascontiguousarray(arr))
                    keepalive.append(arr)
                    bufs[prefix][name[len(prefix) :]] = t
                    break
            else:
                raise KeyError(f".axm 中存在模型不需要的张量: {name}")
        for prefix, module in prefix_map.items():
            module.load_state_dict(bufs[prefix])

    def _prefill_states(
        self,
        pixels,
        grid_t,
        grid_h,
        grid_w,
        ids,
        use_deepstack=True,
    ):
        """公共 prefill: 返回 (h_t, inv_freq_t, pos_t, img_start, want, ds|None)。"""
        device = self.device
        seq = len(ids)
        merge = self.visual.spatial_merge_size
        want = (grid_h // merge) * (grid_w // merge)
        img_start = ids.index(IMAGE_TOKEN_ID)
        img_pos = np.arange(img_start, img_start + want, dtype=np.int64)

        merged, ds = self.visual(pixels, grid_t, grid_h, grid_w, device)
        merged_np = (
            merged.to("cpu").to_numpy() if device != "cpu" else merged.to_numpy()
        )

        ids_t = axono.Tensor.zeros((1, seq), dtype=axono.DataType.INT64, device=device)
        ids_t.copy_from_numpy(np.asarray([ids], dtype=np.int64))
        h = self.text.embed_tokens(ids_t)
        h = _copy(h).reshape((seq, self.text.hidden))
        h_np = h.to("cpu").to_numpy() if device != "cpu" else h.to_numpy()
        for j, ip in enumerate(img_pos):
            h_np[ip] = merged_np[j]
        h_t = _t(h_np, device)

        pos3 = build_mrope_pos3(ids, img_start, want, grid_h // merge, grid_w // merge)
        inv_freq_t = self.text._inv_freq(device)
        pos_t = axono.Tensor.zeros((3, seq), dtype=axono.DataType.INT64, device=device)
        pos_t.copy_from_numpy(np.ascontiguousarray(pos3))

        if not use_deepstack:
            ds = None
        return h_t, inv_freq_t, pos_t, img_start, want, ds

    # ------------------------------------------------------------------
    # KV cache 路径: prefill (整段) + decode (单 token) 两段式
    # ------------------------------------------------------------------
    def create_kv_cache(self, cache=None):
        """创建 KV cache (可传入自定义 nn.KVCache 组件)。"""
        if cache is None:
            cache = nn.DynamicKVCache(self.text.layers_n, device=self.device)
        return cache

    def _run_text_layers(
        self, h_t, seq, inv_freq_t, pos_t, ds, img_start, want, device, cache
    ):
        if ds is not None:
            return self.text.forward_layers_with_deepstack(
                h_t,
                seq,
                inv_freq_t,
                self.text.mrope_section,
                pos_t,
                ds,
                img_start,
                want,
                device,
                cache=cache,
            )
        return self.text.forward_layers(
            h_t,
            seq,
            inv_freq_t,
            self.text.mrope_section,
            pos_t,
            device,
            cache=cache,
        )

    def prefill(
        self,
        pixels: np.ndarray,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        ids,
        deepstack: bool = True,
        kv_cache=None,
    ):
        """整段 prompt 一次前向, k/v 写入 cache; 返回 (last_logits, cache, rope_delta)。

        rope_delta = HF get_rope_index 语义: 图像段 3D 位置比 1D 索引膨胀
        max(h, w)-1, decode 时文本位置 = 绝对索引 + rope_delta。
        """
        cache = self.create_kv_cache(kv_cache)
        device = self.device
        seq = len(ids)
        h_t, inv_freq_t, pos_t, img_start, want, ds = self._prefill_states(
            pixels, grid_t, grid_h, grid_w, ids, deepstack
        )
        h_t = self._run_text_layers(
            h_t, seq, inv_freq_t, pos_t, ds, img_start, want, device, cache
        )
        logits_t = self.text.final_norm_logits(axono.slice_(h_t, 0, seq - 1, 1))
        # rope_delta: 图像段之后文本段的起始 3D 位置 - 图像段末的 1D 索引
        merge = self.visual.spatial_merge_size
        llm_h, llm_w = grid_h // merge, grid_w // merge
        rope_delta = max(llm_h, llm_w) - want
        return logits_t, cache, rope_delta

    def decode_step(self, token_id: int, kv_cache, prev_len: int, rope_delta: int = 0):
        """单 token 解码: 只算新 token, 注意力读 cache 全历史。

        token_id: 上一步选出的 token; prev_len: 本 token 在序列中的 1D 索引
        (即 cache 中已有长度); rope_delta: prefill 返回的 mrope 位置偏移
        (图像段使 3D 位置相对 1D 索引膨胀)。返回该位置 logits (ndarray, vocab)。
        """
        cache = kv_cache
        device = self.device
        seq = 1
        ids_t = axono.Tensor.zeros((1, seq), dtype=axono.DataType.INT64, device=device)
        ids_t.copy_from_numpy(np.asarray([[token_id]], dtype=np.int64))
        h = self.text.embed_tokens(ids_t)
        h_t = _copy(h).reshape((seq, self.text.hidden))

        # HF compute_3d_position_ids (增量路径): position = arange(past_len,
        # past_len+1) + rope_delta, 三个维度同值。
        pos_val = prev_len + rope_delta
        pos_t = axono.Tensor.zeros((3, seq), dtype=axono.DataType.INT64, device=device)
        pos_t.copy_from_numpy(np.full((3, seq), pos_val, dtype=np.int64))
        inv_freq_t = self.text._inv_freq(device)

        h_t = self.text.forward_layers(
            h_t,
            seq,
            inv_freq_t,
            self.text.mrope_section,
            pos_t,
            device,
            cache=cache,
        )
        logits_t = self.text.final_norm_logits(h_t)
        return logits_t.to("cpu").to_numpy() if device != "cpu" else logits_t.to_numpy()

    def forward(
        self,
        pixels: np.ndarray,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        ids,
        deepstack: bool = True,
    ):
        """pixels: (n, 3, t, patch, patch); ids: 含 IMAGE_TOKEN_ID 的文本序列。

        返回 logits ndarray (seq, vocab)。
        """
        device = self.device
        seq = len(ids)
        h_t, inv_freq_t, pos_t, img_start, want, ds = self._prefill_states(
            pixels, grid_t, grid_h, grid_w, ids, deepstack
        )
        if ds is not None:
            h_t = self.text.forward_layers_with_deepstack(
                h_t,
                seq,
                inv_freq_t,
                self.text.mrope_section,
                pos_t,
                ds,
                img_start,
                want,
                device,
            )
        else:
            h_t = self.text.forward_layers(
                h_t, seq, inv_freq_t, self.text.mrope_section, pos_t, device
            )
        logits_t = self.text.final_norm_logits(h_t)
        return logits_t.to("cpu").to_numpy() if device != "cpu" else logits_t.to_numpy()

    def generate(
        self,
        pixels,
        grid_t,
        grid_h,
        grid_w,
        ids,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        greedy: bool = True,
    ):
        """自回归生成 (每步重跑整段前向 — 无 KV cache 的朴素实现)。

        返回 (输出 ids 列表含 prompt, 生成 token 数)。
        """
        ids = list(ids)
        eos_hit = False
        for _ in range(max_new_tokens):
            logits = self.forward(pixels, grid_t, grid_h, grid_w, ids)
            last = logits[-1]
            next_id = int(np.argmax(last))
            if eos_token_id is not None and next_id == eos_token_id:
                eos_hit = True
                break
            ids.append(next_id)
        return ids, eos_hit

    def generate_cached(
        self,
        pixels,
        grid_t,
        grid_h,
        grid_w,
        ids,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        greedy: bool = True,
        kv_cache=None,
    ):
        """KV cache 自回归生成: prefill 一次, 每步只算新 token (O(1) 注意力)。

        kv_cache: 可传入自定义 nn.KVCache 组件; 默认 DynamicKVCache。
        返回 (输出 ids 列表含 prompt, 生成 token 数)。
        """
        ids = list(ids)
        logits_t, cache, rope_delta = self.prefill(
            pixels, grid_t, grid_h, grid_w, ids, kv_cache=kv_cache
        )
        device = self.device
        last = logits_t.to("cpu").to_numpy() if device != "cpu" else logits_t.to_numpy()
        eos_hit = False
        n_prompt = len(ids)
        for _ in range(max_new_tokens):
            next_id = int(np.argmax(last[-1]))
            if eos_token_id is not None and next_id == eos_token_id:
                eos_hit = True
                break
            ids.append(next_id)
            if len(ids) - n_prompt >= max_new_tokens:
                break
            last = self.decode_step(next_id, cache, len(ids) - 1, rope_delta)
        return ids, eos_hit


def load_hf_state_dict(model_dir: str) -> dict:
    """HF safetensors -> {扁平名: fp32 ndarray}。"""
    from safetensors.torch import load_file

    raw = load_file(os.path.join(model_dir, "model.safetensors"))
    return {k: v.float().cpu().numpy() for k, v in raw.items()}
