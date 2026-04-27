# Memory V3 Design

## Goal

`memory_v3` 的目标不是推翻 `v2`，而是在保留 `v2` 可用性的前提下，修复 Qwen3 路线上两个最明显的问题：

1. memory 的真正强项更像是 **用户级校准先验**，而不是完全可靠的边界规则
2. 在缺少相邻分数证据时，`v2` 仍可能写出过强的边界总结，进而在 boundary/fullscale 路线中放大偏置

因此，`v3` 的原则是：

- 保留 `v2` 的核心字段与 cache 可解释性
- 新增程序侧的证据充分性元信息
- 在 prompt 中显式区分 calibration 和 boundary rule
- 在证据不足时强制保守，不把弱证据写成硬规则

## What Stays the Same

`v3` 保留了 `v2` 的主要 LLM 输出字段：

- `avg_satisfaction_score`
- `score_distribution`
- `scoring_style`
- `three_vs_four_distinction`
- `four_vs_five_distinction`
- `user_specific_requirements`
- `preferred_response_format`
- `task_specific_observations`

这样做的目的：

- 保持与 `v2` 的可比性
- 保持历史分析工具和输出结构的连续性
- 避免一次性把太多变量同时改掉

## What Changes in V3

### 1. Program-side evidence sufficiency metadata

`UserMemoryV3` 在 `v2` 基础上增加了以下程序侧字段：

- `calibration_summary`
- `can_compare_3_vs_4`
- `can_compare_4_vs_5`
- `low_score_evidence_level`
- `evidence_notes`

这些字段不是让 LLM 自己编，而是由程序根据 `score_distribution` 直接计算。

这样做的原因是：

- 证据充分性本来就是可判定的统计事实
- 不应再交给模型自由生成

### 2. V3 memory building prompt

`build_memory_prompt_v3()` 的新增约束：

- 会显式告诉模型各分数段样本数量
- 会显式提醒：
  - 若 `3/4` 缺相邻证据，`three_vs_four_distinction` 必须写成弱推断
  - 若 `4/5` 缺相邻证据，`four_vs_five_distinction` 必须写成弱推断
  - 若低分样本极少，不要过度总结 `1/2/3` 细分
- 会要求 `user_specific_requirements` 只保留真正会改变评分的个性化要求，而不是重复“具体、详细、清晰”这类通用偏好

### 3. V3 memory update prompt

`build_memory_update_prompt_v3()` 更强调：

- 优先更新 calibration 信息
- 只有新增 session 提供了明确相邻分数反例时，才修改边界文本
- 不要把原本只是“证据不足”的边界总结，因为单条预测样本就升级为硬规则

### 4. V3 turn-eval prompt

`turn_eval_prompt_version=v3` 是与 `memory_version=v3` 配套的 1-5 打分 prompt。

核心变化：

- 先给模型看 `calibration_summary` 和 score distribution
- 再给模型看边界规则
- 若某个边界被标记为弱推断，prompt 会明确要求不要机械服从该规则
- 当低分证据 sparse/none 时，若回复低于 4 分，默认先收缩到 `3`，只有严重失败时才降到 `2/1`

这条路线的目标是：

- 保留 raw `Qwen none` 中 memory 的校准收益
- 同时降低 boundary/fullscale 中“被弱边界规则带偏”的风险

## Cache Compatibility

为了不覆盖 `v2` cache，`v3` 使用独立缓存文件名：

- `v2`: `{user}__{task}__{model}.json`
- `v3`: `{user}__{task}__{model}__v3.json`

这样可以并行保留：

- 旧 `v2` cache
- 新 `v3` cache

后续实验可直接切换，不会互相污染。

## CLI Changes

新增参数：

- `--memory_version {v2,v3}`

默认仍为：

- `v2`

推荐搭配：

- `memory_version=v3`
- `turn_eval_prompt_version=v3`

当然也可以做交叉验证，例如：

- `memory_version=v3 + turn_eval_prompt_version=v2`

用于判断改动主要来自 memory 还是 prompt。

## Recommended First Run

先做一个小子集对比：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
memory_version=v3 \
turn_eval_prompt_version=v3 \
limit_users=20 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_v3_memv3_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

第一轮建议重点看：

- `MAE / Pearson / QWK`
- block 级 `memory_avg -> pred_mean` 的相关性是否仍然高
- 预测分布是否比 `boundary/fullscale` 更接近真实分布
- `reason` 语义是否保持合法

## Expected Effect

如果 `v3` 方向正确，预期会看到：

1. raw 1-5 预测比 `boundary/fullscale` 更能保留 calibration 优势
2. 对缺少相邻证据的用户，预测不再被虚构边界过度牵引
3. `5 -> 4` 压缩和 SAT/DSAT 单边偏置有所缓和

但短期内不一定会马上显著提高所有指标，因为：

- `v3` 主要是在减少“错误地相信弱证据边界”的问题
- 它更像在提升系统稳健性，而不是直接加更多强监督信号
