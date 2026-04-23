#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate side-by-side BF16 | FP8 PNGs for Stable Diffusion 3.0 / 3.5 (same code path).

Uses the same Omni text-to-image defaults as the offline example (VAE slicing/tiling,
enforce_eager) to reduce VRAM spikes. BF16 and FP8 runs use separate Omni instances so
quantization does not affect the baseline path.

Example:
    pip install pillow
    export VLLM_TARGET_DEVICE=cuda
    export HF_TOKEN=...   # if Hub repos are gated

    python benchmarks/diffusion/sd3_fp8_compare_images.py \\
        --models stabilityai/stable-diffusion-3-medium-diffusers \\
        --output-dir outputs/sd3_fp8_compare

    # Both SD3.0 and SD3.5 (downloads may be large):
    python benchmarks/diffusion/sd3_fp8_compare_images.py --output-dir outputs/sd3_fp8_compare
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path

# Must set before importing vLLM stack in some environments
if "VLLM_TARGET_DEVICE" not in os.environ:
    os.environ["VLLM_TARGET_DEVICE"] = "cuda"

import torch
from PIL import Image, ImageDraw, ImageFont

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


DEFAULT_MODELS = (
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-3.5-medium",
)


def _slug(model: str) -> str:
    s = model.rstrip("/").replace(os.sep, "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120] or "model"


def _extract_pil_image(outputs) -> Image.Image:
    first = outputs[0]
    if hasattr(first, "images") and first.images:
        return first.images[0]
    inner = getattr(first, "request_output", None)
    if inner is not None and hasattr(inner, "images") and inner.images:
        return inner.images[0]
    raise ValueError("Could not extract PIL image from Omni.generate output.")


def _generate_image(
    *,
    model: str,
    quantization: str | None,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    init_timeout: int,
) -> Image.Image:
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=1,
        ring_degree=1,
        ulysses_mode="strict",
        cfg_parallel_size=1,
        tensor_parallel_size=1,
        vae_patch_parallel_size=1,
        enable_expert_parallel=False,
    )
    kwargs: dict = {
        "model": model,
        "mode": "text-to-image",
        "parallel_config": parallel_config,
        "enforce_eager": True,
        "vae_use_slicing": True,
        "vae_use_tiling": True,
        "init_timeout": init_timeout,
    }
    if quantization:
        kwargs["quantization"] = quantization

    omni = Omni(**kwargs)
    try:
        gen = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
        outputs = omni.generate(
            {"prompt": prompt, "negative_prompt": negative_prompt},
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            ),
        )
        return _extract_pil_image(outputs).convert("RGB")
    finally:
        del omni
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _label_strip(img: Image.Image, title: str) -> Image.Image:
    h = 28
    canvas = Image.new("RGB", (img.width, img.height + h), (32, 32, 32))
    canvas.paste(img, (0, h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(240, 240, 240), font=font)
    return canvas


def _concat_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    if left.height != right.height:
        right = right.resize((right.width * left.height // right.height, left.height), Image.Resampling.LANCZOS)
    if left.width != right.width:
        right = right.resize((left.width, left.height), Image.Resampling.LANCZOS)
    w = left.width + right.width + 8
    h = max(left.height, right.height)
    out = Image.new("RGB", (w, h), (24, 24, 24))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 8, 0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="SD3.0/3.5 BF16 vs FP8 side-by-side images.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Hub model id or local diffusers directory (one or more).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sd3_fp8_compare"),
        help="Directory for *_bf16_fp8_compare.png files.",
    )
    parser.add_argument("--prompt", type=str, default="a cup of coffee on a wooden table, morning light")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-timeout", type=int, default=1200, help="Omni init timeout (seconds).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        slug = _slug(model)
        print(f"\n=== Model: {model} ===", flush=True)
        try:
            img_bf16 = _generate_image(
                model=model,
                quantization=None,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                init_timeout=args.init_timeout,
            )
            img_fp8 = _generate_image(
                model=model,
                quantization="fp8",
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                init_timeout=args.init_timeout,
            )
        except Exception as e:
            print(f"[skip] {model}: {e}", file=sys.stderr, flush=True)
            continue

        strip_l = _label_strip(img_bf16, "BF16")
        strip_r = _label_strip(img_fp8, "FP8")
        combined = _concat_side_by_side(strip_l, strip_r)
        out_path = args.output_dir / f"{slug}_bf16_fp8_compare.png"
        combined.save(out_path)
        print(f"Saved: {out_path.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
