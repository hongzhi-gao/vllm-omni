# SD3 / SD3.5 FP8：测试说明

## 配置与 FP8 构建（必跑）

量化配置工厂与 FP8 解析逻辑由下列单测覆盖，**不加载完整 DiT 权重**：

```bash
pytest tests/diffusion/quantization/test_fp8_config.py -q
```

在与本仓库兼容的 **PyTorch 与 vLLM ABI** 下应全部通过（例如 `torch==2.10.0+cu130` 与 `vllm==0.19.0` 配套）。若在 `import torch` 阶段出现 `undefined symbol: ncclDevCommDestroy`，请将 **`nvidia-nccl-cu13`** 升级到与当前 `torch` 发行说明一致的版本（或重装与 `torch` 版本锁定的 `nvidia-nccl-cu13` 轮子）。

**本地记录（示例环境）**：`27 passed`（仅 `test_fp8_config.py`）；与 `tests/test_config_factory.py` 一并执行时为 **`62 passed`**（与 `torch==2.10.0+cu130` + `vllm==0.19.0` ABI 对齐后）。

## 阶段工厂单测（仓库通用）

`StageConfigFactory` 在解析 `parallel_config` 时会对其做 `dataclass.asdict()`；测试中的 `MockParallelConfig` 必须为 **`@dataclass`**，否则会得到 `TypeError: asdict() should be called on dataclass instances`。

```bash
pytest tests/test_config_factory.py -q
```

## 感知质量门（可选 / 重硬件）

`tests/diffusion/quantization/test_quantization_quality.py` 中的 **`fp8_sd3_medium`**（SD3.0 Medium diffusers）与 **`fp8_sd3_5_medium`**（SD3.5 Medium）会：

1. 对同一 prompt 分别用 BF16 与 FP8 各跑一遍 `Omni` 生成；
2. 用 LPIPS 与配置中的 `max_lpips`（当前 **0.25**）比较。

**前置条件：**

- 测试项带 **`@pytest.mark.H100`** 等资源标记：默认 CI 或本地需满足对应 GPU 标记才会执行（否则可能被跳过，取决于项目的 pytest 插件配置）。
- 需能访问 **`stabilityai/stable-diffusion-3-medium-diffusers`** 与/或 **`stabilityai/stable-diffusion-3.5-medium`**（均为 gated 时需配置 **`HF_TOKEN`**，或使用已完整的本地快照路径作为 `model`）。
- 需安装 **`lpips`**（见该测试文件头部说明）。

```bash
pip install lpips
pytest tests/diffusion/quantization/test_quantization_quality.py -k "fp8_sd3_medium or fp8_sd3_5_medium" -v --tb=short
```

若出现 `Could not determine model_type for diffusers model`，说明未能从 Hub 或本地读到带 `_class_name` 的 `model_index.json` / `transformer/config.json`；请先 **`huggingface-cli download`** 到本地，并将 `--model` 指向该目录再跑质量门。

SD3.5 权重较大，若 Hub 下载反复超时，可设置 **`HF_HUB_ENABLE_HF_TRANSFER=1`** 或分时段 **`snapshot_download(..., resume_download=True)`** 直至快照完整；或仅用已就绪的 **SD3.0** 跑 `fp8_sd3_medium`。

## 相关文档

- [FP8 总览与 SD3 矩阵](fp8.md#supported-models)
- [三组对比图命令](sd3_fp8_three_prompts.md)
- [PR 描述模板](sd3_fp8_pr_template.md)
