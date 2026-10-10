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
"""Qwen3-VL-2B 视觉塔 (vision tower) 原生前向 (Axono 算子)。

结构 (单图 T=1):
  patch_embed: Conv3D(t=2,p=16,s=16) == 重排 + matmul + bias
  + 双线性插值 pos_embed (48x48 表 -> h x w 网格)
  24 x [ LayerNorm(1e-6) -> qkv -> 2D axial RoPE -> SDPA(非因果) -> proj + 残差
         -> LayerNorm -> MLP(gelu tanh) + 残差 ]
  merger: LayerNorm -> fc1 -> gelu -> fc2 -> (N/4, 2048)
  deepstack_merger x3 (层 5/11/17)

用法: --device cpu|cuda, 默认与 transformers 视觉塔输出对比 (PASS/FAIL)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import axono

MODEL = "/root/autodl-tmp/qwen3vl-2b"


def load_weights(path: str, device: str) -> dict:
    from safetensors.torch import load_file

    raw = load_file(path)
    w = {}
    for k, v in raw.items():
        arr = v.float().cpu().numpy()
        t = axono.Tensor.from_numpy(np.ascontiguousarray(arr))
        w[k] = t.to(device) if device != "cpu" else t
    return w


def to_t(arr, device):
    t = axono.Tensor.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    return t.to(device) if device != "cpu" else t


def gelu_tanh(x, device):
    """0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3))) — 逐算子组合"""
    arr = x.to_numpy()
    x3 = arr * arr * arr
    inner = np.sqrt(2 / np.pi) * (arr + 0.044715 * x3)
    out = 0.5 * arr * (1.0 + np.tanh(inner))
    return to_t(out, device)


def layer_norm(x, w, b, eps):
    return axono.layer_norm(x, w, b, eps)


def vision_forward(
    w: dict,
    pixels: np.ndarray,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    device: str,
    cfg: dict,
) -> dict:
    """pixels: (grid_t*grid_h*grid_w, 2, 3, 16, 16) float32 patch 序列 (HF 预处理重排后)"""
    v = cfg["vision_config"]
    hidden = v["hidden_size"]
    n_head = v["num_heads"]
    d_head = hidden // n_head
    depth = v["depth"]
    merge = v["spatial_merge_size"]
    eps = 1e-6
    n = grid_t * grid_h * grid_w

    # --- patch embed: conv3d == matmul ---
    x = pixels.reshape(n, -1)  # (N, 2*3*16*16)
    wproj = to_t(
        w["model.visual.patch_embed.proj.weight"].to_numpy().reshape(hidden, -1),
        device,
    )  # conv3d 权重展平 (1024, 512)
    bproj = w["model.visual.patch_embed.proj.bias"]
    x = axono.add(axono.linear_nobias(to_t(x, device), wproj), bproj)

    # --- pos embed 双线性插值 (单图, merge-block 顺序) ---
    side = int(v["num_position_embeddings"] ** 0.5)
    rb, cb = grid_h // merge, grid_w // merge
    r_idx = np.arange(grid_h).reshape(rb, merge)  # (rb, m)
    c_idx = np.arange(grid_w).reshape(cb, merge)  # (cb, m)
    seq = [
        (r_idx[a, ir], c_idx[b, ic])
        for a in range(rb)
        for b in range(cb)
        for ir in range(merge)
        for ic in range(merge)
    ]
    seq = np.array(seq)  # (N, 2) — (row, col) per token, block-major
    rows_i, cols_i = seq[:, 0], seq[:, 1]

    pe = w["model.visual.pos_embed.weight"].to_numpy().reshape(side, side, hidden)
    ys = np.linspace(0, side - 1, grid_h)
    xs = np.linspace(0, side - 1, grid_w)
    pos = np.empty((n, hidden), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows_i, cols_i)):
        ry, rx = ys[r], xs[c]
        iy, ix = min(int(ry), side - 2), min(int(rx), side - 2)
        fy, fx = ry - iy, rx - ix
        pos[i] = (
            pe[iy, ix] * (1 - fy) * (1 - fx)
            + pe[iy, ix + 1] * (1 - fy) * fx
            + pe[iy + 1, ix] * fy * (1 - fx)
            + pe[iy + 1, ix + 1] * fy * fx
        )
    x = axono.add(x, to_t(pos, device))

    # --- 2D axial rope (block-major 位置) ---
    rope_theta = v.get("rope_parameters", {}).get("rope_theta", 10000.0)
    inv_freq = rope_theta ** (
        -np.arange(0, d_head // 2, 2, dtype=np.float32) / (d_head // 2)
    )
    hpos = rows_i.astype(np.float32)
    wpos = cols_i.astype(np.float32)
    fh = hpos[:, None] * inv_freq[None, :]  # (N, d/4)
    fw = wpos[:, None] * inv_freq[None, :]
    fhw = np.concatenate([fh, fw], axis=-1)  # (N, d/2)
    angles = np.concatenate([fhw, fhw], axis=-1)  # (N, d)
    cos_v = np.cos(angles).astype(np.float32)
    sin_v = np.sin(angles).astype(np.float32)

    def rope_vision(t3):
        # t3: (N, heads, d) half-split rotate; cos/sin (N, d) 按 head 广播
        arr = t3.to_numpy().reshape(n, n_head, d_head)
        half = d_head // 2
        c = cos_v[:, None, :]  # (N, 1, d)
        s = sin_v[:, None, :]
        rot = np.concatenate([-arr[..., half:], arr[..., :half]], axis=-1)
        out = arr * c + rot * s
        return to_t(out, device)

    # --- blocks ---
    deepstack = []
    ds_idx = v["deepstack_visual_indexes"]
    import os as _os

    _cap_path = "/tmp/vis_caps.npz"
    _caps = dict(np.load(_cap_path)) if _os.path.exists(_cap_path) else {}
    for li in range(depth):
        p = f"model.visual.blocks.{li}"
        xn1 = layer_norm(x, w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"], eps)
        qkv = axono.add(
            axono.linear_nobias(xn1, w[f"{p}.attn.qkv.weight"]),
            w[f"{p}.attn.qkv.bias"],
        )
        qkv = qkv.to_numpy().reshape(n, 3, n_head, d_head)
        if li == 0 and _caps:
            print(
                "  I in diff:",
                np.abs(x.to_numpy() - _caps["in"].reshape(n, hidden)).max(),
                "n1 diff:",
                np.abs(
                    axono.layer_norm(
                        x, w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"], eps
                    ).to_numpy()
                    - _caps["n1"].reshape(n, hidden)
                ).max(),
                "qkv diff:",
                np.abs(qkv.reshape(n, -1) - _caps["qkv"].reshape(n, -1)).max(),
            )
        q = to_t(qkv[:, 0], device)
        k = to_t(qkv[:, 1], device)
        v_ = to_t(qkv[:, 2], device)
        q = rope_vision(q)
        k = rope_vision(k)
        attn = axono.scaled_dot_product_attention(q, k, v_, False)
        attn = axono.add(
            axono.linear_nobias(attn.reshape((n, hidden)), w[f"{p}.attn.proj.weight"]),
            w[f"{p}.attn.proj.bias"],
        )
        x = axono.add(x, attn)
        xn2 = layer_norm(x, w[f"{p}.norm2.weight"], w[f"{p}.norm2.bias"], eps)
        fc1 = axono.add(
            axono.linear_nobias(xn2, w[f"{p}.mlp.linear_fc1.weight"]),
            w[f"{p}.mlp.linear_fc1.bias"],
        )
        fc2 = axono.add(
            axono.linear_nobias(
                gelu_tanh(fc1, device), w[f"{p}.mlp.linear_fc2.weight"]
            ),
            w[f"{p}.mlp.linear_fc2.bias"],
        )
        x = axono.add(x, fc2)
        if li in ds_idx:
            deepstack.append(x)

    def merger(t, prefix, postshuffle=False):
        hs = hidden * merge * merge
        # HF: x.view(-1, m²*hidden) —— token 已是 merge-block 连续, 直接拼接通道, 无 transpose
        if postshuffle:
            arr = t.to_numpy().reshape(-1, hs)
            t2 = layer_norm(
                to_t(arr, device),
                w[f"{prefix}.norm.weight"],
                w[f"{prefix}.norm.bias"],
                eps,
            )
        else:
            t2 = layer_norm(
                t, w[f"{prefix}.norm.weight"], w[f"{prefix}.norm.bias"], eps
            )
            t2 = to_t(t2.to_numpy().reshape(-1, hs), device)
        fc1 = axono.add(
            axono.linear_nobias(t2, w[f"{prefix}.linear_fc1.weight"]),
            w[f"{prefix}.linear_fc1.bias"],
        )
        # merger 用 nn.GELU() = erf 精确式 (与 blocks 的 tanh 近似不同)
        fc2 = axono.add(
            axono.linear_nobias(axono.gelu(fc1), w[f"{prefix}.linear_fc2.weight"]),
            w[f"{prefix}.linear_fc2.bias"],
        )
        return fc2

    merged = merger(x, "model.visual.merger")
    ds_out = [
        merger(d, f"model.visual.deepstack_merger_list.{i}", True)
        for i, d in enumerate(deepstack)
    ]
    return {"merged": merged, "last_hidden": x, "deepstack": ds_out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()
    device = args.device
    axono.set_backend(device)

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    v = cfg["vision_config"]
    patch = v["patch_size"]
    t = v["temporal_patch_size"]
    # 用 256x256 -> grid 16x16, 单图
    img_h = img_w = 256
    grid_h, grid_w = img_h // patch, img_w // patch
    n = grid_h * grid_w  # 每帧 patch 网格; 时间维打包进每个 patch (t=2 帧)
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal((n, 3, t, patch, patch)).astype(np.float32)
    grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int64)

    print(f"视觉塔前向 ({device}), grid {t}x{grid_h}x{grid_w} ...")
    w = load_weights(os.path.join(args.model, "model.safetensors"), device)
    out = vision_forward(w, pixels, 1, grid_h, grid_w, device, cfg)
    merged = out["merged"].to("cpu").to_numpy()
    print("merged shape:", merged.shape)

    if not args.no_ref:
        import torch
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

        ref = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float32
        ).eval()
        with torch.no_grad():
            r = ref.model.visual(
                torch.from_numpy(pixels.reshape(n, 3 * t, patch, patch)),
                torch.from_numpy(grid_thw),
            )
        ref_merged = r.pooler_output.float().numpy()
        err = np.abs(merged - ref_merged).max()
        print(f"vs transformers merger 输出 max_abs_err = {err:.4e}")
        for i, d in enumerate(out["deepstack"]):
            rd = r.deepstack_features[i].float().numpy()
            print(
                f"deepstack[{i}] err = {np.abs(d.to('cpu').to_numpy() - rd).max():.4e}"
            )
        ok = err < 2e-3
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
