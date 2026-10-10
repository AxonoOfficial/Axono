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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.vlm.qwen3vl.model import (  # noqa: E402
    Qwen3VLForConditionalGeneration,
)

DEFAULT_MODEL = "/root/autodl-tmp/qwen3vl-2b"


class Qwen3VLChat:
    """Qwen3-VL 对话封装: 模型 + processor + 贪心自回归生成。"""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        from transformers import AutoProcessor

        self.max_new_tokens = max_new_tokens
        self.model_dir = model_dir
        device = device or ("cuda" if axono.cuda_available() else "cpu")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration(
            os.path.join(model_dir, "config.json"), device=device
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
        """单轮对话, 返回生成的文本。"""
        ids, pixels, grid = self._encode(prompt, image_path)
        n_prompt = len(ids)
        out = list(ids)
        text_out = []
        t0 = time.time()
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
    chat = Qwen3VLChat(args.model, args.device, args.max_new_tokens)
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
            image_path = prompt[len("/image "):].strip()
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
    args = ap.parse_args()

    if args.device == "cuda" and not axono.cuda_available():
        print("无 CUDA, 回退 CPU")
        args.device = "cpu"

    if args.prompt is not None:
        chat = Qwen3VLChat(args.model, args.device, args.max_new_tokens)
        print("助手> ", end="", flush=True)
        chat.chat(args.prompt, args.image, stream=True)
        return 0

    interactive(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
