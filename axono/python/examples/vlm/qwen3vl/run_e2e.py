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
"""Qwen3-VL Module 版端到端数值验证 (vs HF transformers)。

用法: PYTHONPATH=../.. python -m examples.vlm.qwen3vl.run_e2e --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.vlm.qwen3vl.model import (  # noqa: E402
    IMAGE_TOKEN_ID,
    Qwen3VLForConditionalGeneration,
)

DEFAULT_MODEL = "/root/autodl-tmp/qwen3vl-2b"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()

    print(f"初始化 Qwen3VLForConditionalGeneration ({args.device}) ...")
    model = Qwen3VLForConditionalGeneration(
        os.path.join(args.model, "config.json"), device=args.device
    )
    print("加载 HF safetensors 权重 ...")
    model.load_hf_weights(args.model)
    print(f"  参数量: {len(model.parameters())}")

    v = model.config["vision_config"]
    patch = v["patch_size"]
    grid_h = grid_w = 256 // patch
    n = grid_h * grid_w
    want = n // (v["spatial_merge_size"] ** 2)
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal(
        (n, 3, v["temporal_patch_size"], patch, patch)
    ).astype(np.float32)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    ids = tok.encode("<|im_start|>user\n").ids
    ids = ids + [IMAGE_TOKEN_ID] * want
    ids = ids + tok.encode("描述这张图片<|im_end|>\n<|im_start|>assistant\n").ids
    print(f"input_ids: {len(ids)} tokens, image tokens: {want}")

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
    ok = err < 5e-2 and np.array_equal(
        np.argsort(-last)[:1], np.argsort(-ref_lg)[:1]
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
