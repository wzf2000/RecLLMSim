# `boundary_34_refute` Prompt 设计说明

## 目标

在现有 `boundary_34` 基础上，新增一个更适合 Qwen3-8B 的 `3/4` 边界 prompt 版本：

- 名称：`boundary_34_refute`
- 输出仍然只允许 `3` 或 `4`
- 保留现有流程和历史版本，不替换旧版本

## 为什么要做这个版本

`boundary_34` 的结果表明，单纯把输出空间收缩到 `3/4` 不够，模型仍然会强烈偏向：

- “只要回复看起来在帮忙，就给 4”

其直接后果是：

- `F1-DSAT` 下降
- `false_sat_rate` 明显升高
- 用户级边界能力退化

说明问题不在于“模型还在区分 1/2/5”，而在于它缺少一个更强的、面向负例的检查过程。

## 设计思路

`boundary_34_refute` 的核心变化不是继续压缩输出，而是把推理顺序改成：

1. **先找足以判成 3 的关键失败**
2. **做反证检查**
3. **只有在反证失败时，才允许给 4**

也就是把原先的：

- “有没有达到最低满意线？”

改成更严格的：

- “有没有任何一个明确证据表明它其实没达到最低满意线？”

## 关键设计点

### 1. failure-first

新版 prompt 显式要求先检查这些足以降到 3 分的关键失败：

- 没有直接回答主要问题
- 忽略关键约束或任务目标
- 内容过于泛化，缺乏可执行性
- 漏掉用户特定要求
- 存在会明显伤害满意度的实质性缺口

### 2. refutation step

新版要求模型先问自己：

- “如果我要把它判成 3，最强证据是什么？”

如果存在明确、实质的证据，就直接判 `3`；只有这些证据都不成立时，才允许判 `4`。

### 3. anchor 用法改成“找失败模式”

如果启用 anchor，旧版更容易把案例当成模糊参照；新版要求：

- 优先看 `<=3` 的案例缺了什么
- 再看 `>=4` 的案例满足了什么

也就是让 anchor 更像“边界失败模式库”，而不是简单的“相似示例”。

## 预期作用

这版的目标不是提高全局 MAE，而是更有针对性地改善：

- `F1-DSAT`
- `false_sat_rate`
- `PU-F1-DSAT`
- `PU-Kappa`
- `WC-bin-Pearson`

尤其是希望降低 `boundary_34` 中“默认判满意”的 SAT 偏置。

## 运行方式

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_refute \
bash scripts/collect_personalized_vllm.sh
```

输出文件会自动带上 `_boundary_34_refute` 后缀。

## 稳定性补充

在首轮运行中，`boundary_34_refute` 相比旧版本更容易触发结构化解析失败。为降低这类错误，当前实现额外做了两点收敛：

- **单独降温**：`boundary_34_refute` 的 turn prediction 温度从通用的 `0.6` 降到 `0.2`
- **专用 schema**：边界版本不再复用通用 `1-5` schema，而是改为只允许 `classification ∈ {3, 4}`

此外，边界版本的解析路线也做了分流：

- `v2 / qwen_short`：继续走原来的 SDK `.parse()` 路线
- `boundary_34 / boundary_34_refute`：改为普通 `chat.completions.create()`，然后在本地做：
  - 去除 `<think>...</think>` / `</think>`
  - 去除 code fence
  - 容错提取 `classification / reason / analysis`
  - 再用本地 schema 校验

这样做的原因是：Qwen3-8B 在 vLLM 下偶发会生成“几乎正确但不是合法 JSON”的结果；若继续直接依赖 SDK `.parse()`，会在 SDK 层先抛错，来不及进入本地恢复逻辑。

同时，若解析失败，会将调试信息写入：

- `outputs/personalized/parse_failures/*.json`
- `outputs/personalized/parse_failures/*.prompt.txt`
- `outputs/personalized/parse_failures/*.raw.txt`

便于后续检查失败 turn 的 `block_id / turn_idx / prompt_version / prompt_length / error`。
