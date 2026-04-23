# PR 描述模板：SD3 DiT 动态 FP8（复制到 GitHub PR）

**已填好实测表与对比图路径的正文**见：[PR_SD3_FP8.md](PR_SD3_FP8.md)（推荐直接复制该文件）。  
本节保留为精简模板；将占位符替换为实测值与 PR 编号：`#____`、`BF16_*`、`FP8_*`、`GPU_NAME`。

## 1. Motivation

Stable Diffusion 3 DiT 在 `--quantization fp8` 下使用**在线动态 FP8**，降低权重与部分前向显存占用；对注意力与上下文支路等敏感路径保持 BF16，与 Hunyuan 等对 attention 的保守策略一致。

## 2. Scope

- **仅 DiT（transformer）**：文本编码器与 VAE 仍为 FP32/BF16，与默认管线一致。
- **BF16 路径**：`diffuse` 去噪循环与加入 FP8 显存技巧前的行为一致（无额外 `empty_cache` / `del` 中间逻辑）。

## 3. Implementation summary

- DiT：注意力、`context_embedder`、`proj_out`、context 支路 `ff_context` 保持 BF16；图像支路 `ff` 仅 **down-proj** 走 FP8（`quantize_down_proj_only`）。
- FP8：`quant_config_is_fp8` 供 pipeline 判断；`_contiguous_if_needed` 避免无谓 contiguous 拷贝。
- FP8：encode 完成后将文本塔迁 CPU，并依次调用 `current_omni_platform.empty_cache()` 与 `torch.cuda.empty_cache()`（经 `_sd3_fp8_maybe_empty_cuda_cache`）。
- 去噪结束后删除条件 embedding 并 `empty_cache`（BF16/FP8 共用，与 Wan2.2 思路一致）。
- FP8：VAE decode 后 `del latents` 并 `_sd3_fp8_maybe_empty_cuda_cache`。
- FP8：DiT 在**全部** `transformer_blocks` 结束后**一次** `torch.cuda.empty_cache()`（不再按每 8 个 block 调用）。

## 4. VRAM

在同一 GPU（`GPU_NAME`）与相同生成参数下，从日志读取 **Model loading** 与 **Peak GPU memory**（reserved / allocated）峰值，填入下表。

| 指标 | BF16 | FP8 | 节省 |
|------|------|-----|------|
| Model load (GiB) | BF16_model_load | FP8_model_load | BF16 − FP8 |
| Peak reserved (GB) | BF16_peak_reserved | FP8_peak_reserved | BF16 − FP8 |
| Peak allocated (GB) | BF16_peak_allocated | FP8_peak_allocated | BF16 − FP8 |

说明：显存受驱动与 PyTorch 缓存分配器影响会有小幅波动，表中数字与单次运行日志一致即可。

## 5. Quality

- 同 prompt / seed 的 BF16 与 FP8 对比图：见 PR 附件或上传链接（建议三组：风景 / 人像 / 静物）。
- 输出与 BF16 **非 bit-identical** 为预期；与 `empty_cache` 频率无直接关系。

## 6. Reproduce

依赖：`examples/offline_inference/text_to_image/text_to_image.py` 或 `benchmarks/diffusion/sd3_fp8_compare_images.py`。模型任选 **`stabilityai/stable-diffusion-3-medium-diffusers`（SD3.0）** 或 **`stabilityai/stable-diffusion-3.5-medium`（SD3.5）**；`768×768`、`28` 步、`--guidance-scale 4.5`、`--vae-use-slicing --vae-use-tiling`、`--enforce-eager`，并设置相同 `--seed` 与 `--negative-prompt`。

**三组题材（风景 / 人像 / 静物）** 的完整命令与示例 prompt 见：[sd3_fp8_three_prompts.md](sd3_fp8_three_prompts.md)。

**BF16：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model stabilityai/stable-diffusion-3.5-medium \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "YOUR_NEGATIVE" \
  --prompt "YOUR_PROMPT" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /path/out_bf16.png
```

**FP8：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model stabilityai/stable-diffusion-3.5-medium \
  --quantization fp8 \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "YOUR_NEGATIVE" \
  --prompt "YOUR_PROMPT" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /path/out_fp8.png
```

## 7. Testing

必跑与可选项、NCCL / ABI 说明见仓库文档：[sd3_fp8_testing.md](sd3_fp8_testing.md)。

```bash
pytest tests/diffusion/quantization/test_fp8_config.py -q
pytest tests/test_config_factory.py -q
```

`fp8_sd3_medium` / `fp8_sd3_5_medium`（LPIPS 质量门）需 H100 等资源标记、HF 模型可访问性及 `lpips`；详见 [sd3_fp8_testing.md](sd3_fp8_testing.md)。

## 8. Docs

- 更新 [FP8 支持模型表](fp8.md#supported-models) 与 [量化总览](overview.md) 中 FP8 行。
- 功能矩阵：[diffusion_features.md](../../diffusion_features.md) 中 **Stable-Diffusion3 (3.0 & 3.5)** 的 **Quantization** 列为 ✅（与本 PR 一致）。

## Quantization Matrix（GitHub 表格用，PR #____）

| Model | Type | Attn | MLP | Notes |
|-------|------|:----:|:---:|-------|
| StableDiffusion3 / 3.5 (DiT) | D | ❌ | ✅ #____ | Image FFN down-proj only (online FP8); context FFN & projections BF16; text encoders / VAE BF16 |
