# SD3 / SD3.5：三组 BF16 vs FP8 对比图（可复现命令）

在已激活的 venv 下，从仓库根目录执行。`--model` 请指向**含完整 `model_index.json` 的 diffusers 目录**（Hub ID 或本地快照均可）。若仅缓存了 README、权重未下全，请改用已完整的 **SD3 medium** 快照，例如：

`/home/hongzhi/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671`

下文命令里将上述路径写作 `$SD3_MODEL_DIR`，导出一次即可：

```bash
export SD3_MODEL_DIR="/home/hongzhi/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
```

**SD3.5 Medium**：将 `$SD3_MODEL_DIR` 设为 Hub ID **`stabilityai/stable-diffusion-3.5-medium`**（或已下载的本地快照目录）即可，命令行参数与下文相同；与 SD3.0 共用同一套 `text_to_image.py` 与 FP8 实现。

**一键并排对比图**（BF16 | FP8，脚本内两次独立 `Omni`）：见仓库根目录下 `python benchmarks/diffusion/sd3_fp8_compare_images.py --help`。

统一参数：`768×768`、`28` 步、`--guidance-scale 4.5`、`--seed 42`、`--vae-use-slicing --vae-use-tiling`、`--enforce-eager`。

输出目录优先：`/mnt/d/sd3_fp8_compare/`（便于 WSL 拷贝到 Windows）；也可改为任意可写路径。

**共用负向提示（示例）：**

```text
low quality, worst quality, blurry, jpeg artifacts, bad anatomy, deformed, text, watermark
```

## 1. 风景

**Prompt（示例）：**

```text
A serene mountain lake at golden hour, pine forest, reflection on water, dramatic clouds, highly detailed, photorealistic
```

**BF16：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "$SD3_MODEL_DIR" \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, deformed, text, watermark" \
  --prompt "A serene mountain lake at golden hour, pine forest, reflection on water, dramatic clouds, highly detailed, photorealistic" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /mnt/d/sd3_fp8_compare/theme1_landscape_bf16.png
```

**FP8：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "$SD3_MODEL_DIR" \
  --quantization fp8 \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, deformed, text, watermark" \
  --prompt "A serene mountain lake at golden hour, pine forest, reflection on water, dramatic clouds, highly detailed, photorealistic" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /mnt/d/sd3_fp8_compare/theme1_landscape_fp8.png
```

## 2. 人像

**Prompt（示例）：**

```text
Portrait of a woman in soft window light, natural skin texture, 85mm lens, shallow depth of field, neutral background, editorial photography
```

**BF16：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "$SD3_MODEL_DIR" \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, deformed, text, watermark" \
  --prompt "Portrait of a woman in soft window light, natural skin texture, 85mm lens, shallow depth of field, neutral background, editorial photography" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /mnt/d/sd3_fp8_compare/theme2_portrait_bf16.png
```

**FP8：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "$SD3_MODEL_DIR" \
  --quantization fp8 \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, deformed, text, watermark" \
  --prompt "Portrait of a woman in soft window light, natural skin texture, 85mm lens, shallow depth of field, neutral background, editorial photography" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /mnt/d/sd3_fp8_compare/theme2_portrait_fp8.png
```

## 3. 静物

**Prompt（示例）：**

```text
Still life of ceramic vase with dried flowers on wooden table, soft studio lighting, muted colors, high detail, product photography
```

**BF16：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "$SD3_MODEL_DIR" \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, deformed, text, watermark" \
  --prompt "Still life of ceramic vase with dried flowers on wooden table, soft studio lighting, muted colors, high detail, product photography" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /mnt/d/sd3_fp8_compare/theme3_stilllife_bf16.png
```

**FP8：**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "$SD3_MODEL_DIR" \
  --quantization fp8 \
  --height 768 --width 768 \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --negative-prompt "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, deformed, text, watermark" \
  --prompt "Still life of ceramic vase with dried flowers on wooden table, soft studio lighting, muted colors, high detail, product photography" \
  --seed 42 \
  --vae-use-slicing --vae-use-tiling \
  --enforce-eager \
  --output /mnt/d/sd3_fp8_compare/theme3_stilllife_fp8.png
```

## 显存摘录

每次运行从控制台或日志中查找：

- `Model loading took … GiB`
- `Peak GPU memory (this request): … reserved, … allocated`

填入 PR 表格（见 [sd3_fp8_pr_template.md](sd3_fp8_pr_template.md)）。
