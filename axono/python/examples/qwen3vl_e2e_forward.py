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
"""Qwen3-VL-2B 端到端前向: Axono 视觉塔 + 文本塔 (deepstack + M-RoPE) vs transformers。

用法: --device cpu|cuda (默认 cpu), 与 transformers 完整多模态 logits 对比。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import axono

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "/root/autodl-tmp/qwen3vl-2b"
IMAGE_TOKEN_ID = 151655


def load_all_weights(path: str, device: str) -> dict:
    from safetensors.torch import load_file

    raw = load_file(path)
    w = {}
    for k, v in raw.items():
        arr = v.float().cpu().numpy()
        t = axono.Tensor.from_numpy(np.ascontiguousarray(arr))
        w[k] = t.to(device) if device != "cpu" else t
    return w


def build_mrope_pos3(ids, img_pos, want, llm_h, llm_w):
    """HF get_rope_index 约定 (单图): 文本段 3 维 = 顺序计数;
    图像段 = llm 网格 raster (T=start, H=start+h, W=start+w); 段后从 start+max(h,w) 继续。"""
    n = len(ids)
    img_start = int(img_pos[0])
    pos3 = np.zeros((3, n), dtype=np.int64)
    pos3[:, :img_start] = np.arange(img_start)[None, :]
    cur = img_start
    t_seg = np.arange(img_start, img_start + want)
    pos3[0, t_seg] = cur
    pos3[1, t_seg] = cur + np.arange(want) // llm_w
    pos3[2, t_seg] = cur + np.arange(want) % llm_w
    cur += max(llm_h, llm_w)
    n_tail = n - (img_start + want)
    pos3[:, img_start + want :] = (np.arange(n_tail) + cur)[None, :]
    return pos3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()
    device = args.device

    axono.set_backend("cuda" if device == "cuda" else "cpu")

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    vcfg = cfg["vision_config"]
    patch = vcfg["patch_size"]
    img = 256
    grid_h = grid_w = img // patch
    n = grid_h * grid_w
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal((n, 3, 2, patch, patch)).astype(np.float32)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    want = n // 4  # 合并后 image token 数
    ids = tok.encode("<|im_start|>user\n").ids
    ids = ids + [IMAGE_TOKEN_ID] * want
    ids = ids + tok.encode("描述这张图片<|im_end|>\n<|im_start|>assistant\n").ids
    img_pos = np.arange(
        ids.index(IMAGE_TOKEN_ID), ids.index(IMAGE_TOKEN_ID) + want, dtype=np.int64
    )
    print(f"input_ids: {len(ids)} tokens, image tokens: {want}")

    print(f"端到端前向 ({device}), grid 1x{grid_h}x{grid_w} ...")
    w = load_all_weights(os.path.join(args.model, "model.safetensors"), device)

    from qwen3vl_text_forward import qwen3_text_forward  # noqa: E402
    from qwen3vl_vision_forward import vision_forward  # noqa: E402

    vis_out = vision_forward(w, pixels, 1, grid_h, grid_w, device, cfg)
    merged = vis_out["merged"]
    deepstack = vis_out["deepstack"]

    # embedding + image token 替换
    emb = w["model.language_model.embed_tokens.weight"]
    ids_t = axono.Tensor.zeros((1, len(ids)), dtype=axono.DataType.INT64, device=device)
    ids_t.copy_from_numpy(np.asarray([ids], dtype=np.int64))
    h = axono.embedding(ids_t, emb).to_numpy()[0]  # (L, hidden)
    m_np = merged.to("cpu").to_numpy() if device != "cpu" else merged.to_numpy()
    for j, ip in enumerate(img_pos):
        h[ip] = m_np[j]
    hidden = h

    merge = vcfg["spatial_merge_size"]
    pos3 = build_mrope_pos3(ids, img_pos, want, grid_h // merge, grid_w // merge)

    logits = qwen3_text_forward(
        w,
        ids,
        device,
        cfg,
        hidden=hidden,
        deepstack=deepstack,
        image_positions=img_pos,
        mrope_pos3=pos3,
    )
    last = logits[-1] if logits.ndim == 2 else logits[0, -1]
    print("Top-5:", np.argsort(-last)[:5].tolist())

    if args.no_ref:
        return 0

    import torch
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

    ref = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).eval()
    pixel_values = torch.from_numpy(pixels.reshape(n, 3 * 2, patch, patch))
    grid_thw = torch.tensor([[1, grid_h, grid_w]])
    input_ids = torch.tensor([ids])
    mm_ids = torch.zeros_like(input_ids)
    mm_ids[input_ids == IMAGE_TOKEN_ID] = 1
    with torch.no_grad():
        r = ref(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            mm_token_type_ids=mm_ids,
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
