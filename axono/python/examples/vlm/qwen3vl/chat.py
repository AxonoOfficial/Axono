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
"""Qwen3-VL-2B 完整对话 (多模态) — 基于 Axono nn.Module 前向。

用 HF AutoProcessor 做 chat template 与图像预处理, 前向/生成全部走 Axono。

交互式:
    PYTHONPATH=. python -m examples.vlm.qwen3vl.chat --device cuda --model /path/to/Qwen3-VL-2B-Instruct

单次:
    PYTHONPATH=. python -m examples.vlm.qwen3vl.chat --device cuda --model /path \
        --prompt "描述这张图片" --image /path/to/img.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import axono

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from examples.vlm.qwen3vl.model import (  # noqa: E402
    Qwen3VLForConditionalGeneration,
)

DEFAULT_MODEL = "/root/autodl-tmp/qwen3vl-2b"


class Qwen3VLChat:
    """Qwen3-VL 对话封装: 模型 + processor + 贪心自回归生成。

    use_kv_cache: 开启 KV cache (默认开; prefill 一次, 每步只算新 token)。
    kv_cache_factory: 自定义 KV Cache 组件工厂 () -> nn.KVCache;
        传入后优先于默认 DynamicKVCache。
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 256,
        use_kv_cache: bool = True,
        kv_cache_factory=None,
        use_gqa: bool = True,
    ):
        from transformers import AutoProcessor

        self.max_new_tokens = max_new_tokens
        self.model_dir = model_dir
        self.use_kv_cache = use_kv_cache
        self.kv_cache_factory = kv_cache_factory
        self.use_gqa = use_gqa and axono.cuda_available()
        device = device or ("cuda" if axono.cuda_available() else "cpu")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration(
            os.path.join(model_dir, "config.json"),
            use_gqa=self.use_gqa,
            device=device,
        )
        print(f"加载权重 ({device}) ...")
        self.model.load_hf_weights(model_dir)
        self.model.eval()
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    # -- 输入编码: 文本 + 可选单图 --
    def _encode(self, prompt: str, image_path: str | None):
        content = []
        image = None
        if image_path is not None:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if image is not None:
            inputs = self.processor(text=[text], images=[image], return_tensors="np")
        else:
            inputs = self.processor(text=[text], return_tensors="np")

        ids = inputs["input_ids"][0].tolist()
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"].astype(np.float32)
            grid = inputs["image_grid_thw"][0].tolist()
        else:
            pv, grid = None, None
        return ids, pv, grid

    def _decode_step(self, ids, pixels, grid):
        """一步前向 (无 KV cache: 重跑整段), 返回最后一个位置的 logits。"""
        if pixels is None:
            raise ValueError("纯文本对话尚未支持: 请提供 --image")
        gt, gh, gw = grid
        logits = self.model.forward(pixels, int(gt), int(gh), int(gw), ids)
        return logits[-1]

    def chat(self, prompt: str, image_path: str | None = None, stream: bool = True):
        """单轮对话, 返回生成的文本。

        KV cache 开启时: prefill 一次 + 逐 token decode (流式输出);
        关闭时: 朴素逐步整段重算 (与旧版行为一致, 可用于对照)。
        """
        ids, pixels, grid = self._encode(prompt, image_path)
        n_prompt = len(ids)
        out = list(ids)
        text_out = []
        t0 = time.time()

        if self.use_kv_cache:
            if pixels is None:
                raise ValueError("纯文本对话尚未支持: 请提供 --image")
            gt, gh, gw = (int(v) for v in grid)
            cache = (
                self.kv_cache_factory() if self.kv_cache_factory is not None else None
            )
            logits_t, cache, rope_delta = self.model.prefill(
                pixels, gt, gh, gw, ids, kv_cache=cache
            )
            last = (
                logits_t.to("cpu").to_numpy()
                if self.device != "cpu"
                else logits_t.to_numpy()
            )
            for step in range(self.max_new_tokens):
                nxt = int(np.argmax(last[-1]))
                if nxt == self.eos_token_id or nxt == self.im_end_id:
                    break
                out.append(nxt)
                piece = self.processor.tokenizer.decode([nxt], skip_special_tokens=True)
                text_out.append(piece)
                if stream:
                    print(piece, end="", flush=True)
                if step + 1 < self.max_new_tokens:
                    last = self.model.decode_step(nxt, cache, len(out) - 1, rope_delta)
        else:
            for step in range(self.max_new_tokens):
                last = self._decode_step(out, pixels, grid)
                nxt = int(np.argmax(last))
                if nxt == self.eos_token_id or nxt == self.im_end_id:
                    break
                out.append(nxt)
                piece = self.processor.tokenizer.decode([nxt], skip_special_tokens=True)
                text_out.append(piece)
                if stream:
                    print(piece, end="", flush=True)
        if stream:
            print()
        dt = time.time() - t0
        n_new = len(out) - n_prompt
        if n_new:
            print(
                f"[{n_new} tokens, {dt:.2f}s, {n_new / dt:.1f} tok/s, "
                f"prompt {n_prompt} tokens]"
            )
        return "".join(text_out)


def interactive(args):
    chat = Qwen3VLChat(
        args.model,
        args.device,
        args.max_new_tokens,
        use_kv_cache=not args.no_kv_cache,
        use_gqa=not args.no_gqa,
    )
    print("进入交互模式 (输入 quit 退出; 用 /image <path> 设置图片)。\n")
    image_path = args.image
    if image_path:
        print(f"当前图片: {image_path}")
    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt in ("quit", "exit", "/quit"):
            break
        if prompt.startswith("/image "):
            image_path = prompt[len("/image ") :].strip()
            print(f"已设置图片: {image_path}")
            continue
        print("助手> ", end="", flush=True)
        chat.chat(prompt, image_path, stream=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default=None, help="单次问答; 省略则进交互模式")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--no-gqa",
        action="store_true",
        help="关闭 GQA decode 优化 kernel (回退通用 SDPA)",
    )
    ap.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="关闭 KV cache (每步重跑整段前向, 与旧版行为一致)",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not axono.cuda_available():
        print("无 CUDA, 回退 CPU")
        args.device = "cpu"

    if args.prompt is not None:
        chat = Qwen3VLChat(
            args.model,
            args.device,
            args.max_new_tokens,
            use_kv_cache=not args.no_kv_cache,
            use_gqa=not args.no_gqa,
        )
        print("助手> ", end="", flush=True)
        chat.chat(args.prompt, args.image, stream=True)
        return 0

    interactive(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
