# Qwen3.6 Local Migration Feasibility

## Context

Previous local validation used vLLM with Qwen3-8B and a 32,768-token context
window. The next candidate models are Qwen3.6-35B-A3B and Qwen3.6-27B,
because model scale may affect the conclusions of personalized satisfaction
detection experiments.

## Local Environment Snapshot

Collected on 2026-05-09 from `/data/wangzhefan/RecLLMSim`.

- `nvidia-smi` could not communicate with the NVIDIA driver.
- `/dev/nvidia*` devices were not visible in the current shell.
- PyTorch reported `gpu_count=0` and warned that NVML could not initialize.
- Python environment: torch `2.10.0+cu128`, CUDA runtime `12.8`, vLLM `0.18.1`.
- Host memory: 2.0 TiB total, 1.9 TiB available.
- Workspace disk: 14T total, 5.4T available.
- `/tmp`: 40G total, 6.9G available.

The current shell therefore cannot run either candidate through CUDA until the
GPU driver/device visibility issue is resolved.

The user-provided `nvidia-smi` output from 2026-05-09 18:45 shows the underlying
machine has 8x NVIDIA A100-SXM4-80GB GPUs with driver `550.144.03` and CUDA
`12.4`. GPU 3, GPU 4, and GPU 5 were effectively idle with only 1 MiB memory
used. GPU 0/1 were occupied by a TP2 vLLM worker pair, GPU 2 was almost full and
reported 11 volatile uncorrectable ECC events, and GPU 6/7 were busy with Python
processes.

This means the host is suitable for larger-model validation, but the current
Codex shell lacks GPU device visibility. Experiments should be launched from an
environment where `nvidia-smi` and `torch.cuda.device_count()` can see the A100s.

## Local Model Cache

- `Qwen/Qwen3-8B`: about 16G cached.
- `Qwen/Qwen3.6-35B-A3B`: about 67G cached; safetensor shards are present via
  Hugging Face cache symlinks.
- `Qwen/Qwen3.6-27B`: about 404M cached; only config/template/license style
  files are present, so model weights still need to be downloaded.

## Model Notes

Based on the official Hugging Face model cards and vLLM recipe:

- Qwen3.6-35B-A3B is a sparse MoE model with 35B total parameters and 3B active
  parameters. It has a native 262,144-token context. The vLLM recipe lists
  FP8 serving as suitable for a single H100/H200-class GPU, and BF16 serving as
  suitable for 1x H200 or 2x H100 for full 262K context.
- Qwen3.6-27B is a dense 27B model with a native 262,144-token context. The
  model card recommends vLLM `>=0.19.0` and shows 8-way tensor parallel serving
  for full 262K context.

For the current experiment's 32,768-token context, memory pressure is much lower
than 262K, but model weights still dominate the footprint.

## Recommendation

Do not switch blindly from Qwen3-8B to only one larger model. Use a staged
comparison:

1. First repair GPU visibility and run a one-batch smoke test.
2. Prefer Qwen3.6-35B-A3B as the first upgrade candidate on idle A100s 3/4/5,
   because the 3B-active MoE design should be cheaper per generated token than
   dense 27B while still increasing model capacity. On A100, use BF16 first;
   the official single-GPU FP8 recommendation targets H100/H200-class hardware.
3. Start Qwen3.6-35B-A3B with TP2 on GPU 3 and GPU 4 at 32K context. If memory
   is tight, reduce concurrency or use TP3 on GPUs 3/4/5.
4. Use Qwen3.6-27B as the stronger dense-model validation point only after
   confirming enough GPU memory or tensor parallel resources. It may be more
   expensive and slower, and the local weights are not currently downloaded.
5. Keep Qwen3-8B as the baseline and rerun a fixed subset across all models
   before reinterpreting earlier conclusions.

For vLLM settings, start with text-only serving if the experiments do not use
images:

- `--max-model-len 32768`
- `--language-model-only`
- `--reasoning-parser qwen3`
- low concurrency first, then increase after measuring peak GPU memory

For Qwen3.6-27B, upgrade vLLM to at least `0.19.0` in a separate environment
before drawing conclusions from failures or performance.
