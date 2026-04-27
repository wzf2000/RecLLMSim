# Qwen3-8B 短 Prompt 实验设计

## 背景

现有 `memory v2` 的 turn evaluation prompt 对 GPT-4o-mini 效果较好，但对 Qwen3-8B 仍存在两个明显问题：

- 绝对分值校准较弱，尤其容易将真实 5 分压到 4 分
- 长 prompt 中信息较多，Qwen3-8B 未必能稳定利用全部 memory 字段

因此新增一个**保留原逻辑但更短、更硬的 Qwen 专用 prompt**，作为后续对照实验版本。

## 新增版本

- 原版：`turn_eval_prompt_version=v2`
- 新版：`turn_eval_prompt_version=qwen_short`

默认仍使用 `v2`，保证历史实验可复现；只有显式传参时才切换到 `qwen_short`。

## 设计原则

- 保留 memory v2 的核心结构：`3分以下→4分门槛`、`4分→5分门槛`、`user_specific_requirements`
- 压缩指令长度，减少冗长解释
- 把推理步骤硬化成 3 步 checklist：
  1. 先判是否达到 4 分基线
  2. 若达到，再判是否满足 5 分门槛
  3. 选择最贴切原因标签
- 强制 analysis 简短，只保留与最终打分直接相关的依据

## 预期收益

- 提升 Qwen3-8B 对 memory 字段的利用率
- 降低 5→4 的系统性压分
- 为后续 `4/5 二次判别`、`对比式 anchor` 等实验提供更干净的起点

## 运行方式

vLLM 服务：

```bash
cd detection
model=Qwen/Qwen3-8B bash scripts/serve_vllm.sh
```

收集 trace：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=qwen_short \
bash scripts/collect_personalized_vllm.sh
```

输出文件会自动带上 `_qwen_short` 标签，便于与原版 `v2` 对照。
