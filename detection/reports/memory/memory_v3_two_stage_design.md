# Memory V3 Two-Stage Design

## Goal

`v3` 和 `v3.1` 的结果说明：

- calibration 分离是有价值的
- 但单 prompt 里同时做 calibration、SAT gate、`4/5` 细分、`1/2/3` 细分，仍然容易互相干扰

因此新增 `v3_two_stage`：

1. Stage 1：只判是否通过最低满意线（`3/4` gate）
2. Stage 2A：若通过，再细分 `4/5`
3. Stage 2B：若未通过，再细分 `1/2/3`

## Main Design Principle

核心原则是：

- calibration 不再参与“是否满意”的最终裁决
- calibration 只作为背景刻度信息
- `SAT gate` 必须先独立完成

这比 `v3.1` 更进一步，因为它不再尝试在同一个 prompt 里完成所有层次判断。

## Stage 1: SAT Gate

第一层只输出：

- `4` = 通过最低满意线
- `3` = 未通过最低满意线

判断重点：

- 核心问题是否被直接回答
- 关键约束 / 关键任务目标 / 用户特别在意的要求是否被满足
- 剩余问题是否只是普通缺口，而不是关键失败

如果 `3/4` 边界证据不足：

- 不会放松 gate
- 只会更依赖上述三个问题与真实案例

## Stage 2A: SAT Refinement

如果第一层判为 SAT：

- 第二层只在 `4/5` 之间细分
- 默认先给 `4`
- 若 `4/5` 边界证据不足，继续默认保守给 `4`

## Stage 2B: DSAT Refinement

如果第一层判为 DSAT：

- 第二层只在 `1/2/3` 之间细分
- 默认先给 `3`
- 若低分证据 sparse/none，只有严重不可用、明显错误、严重答非所问时才降到 `2/1`

## Output Fields

该版本会额外写出：

- `analysis_gate`
- `analysis_sat_refine`
- `analysis_dsat_refine`
- `two_stage_gate_score`
- `two_stage_gate_reason`
- `two_stage_gate_analysis`
- `two_stage_branch`
- `two_stage_refine_applied`

这样后续可以直接拆开看：

- gate 是否是瓶颈
- SAT 分支是否过度压 `5`
- DSAT 分支是否仍然几乎不给 `1/2`

## Recommended Run

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
memory_version=v3 \
turn_eval_prompt_version=v3_two_stage \
limit_users=20 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_two_stage_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

## What to Watch

这版最关键的观察点：

- 相比 `v3.1`，`F1-DSAT` 是否明显回升
- `false_sat_rate` 是否明显下降
- `MAE / Pearson / Spearman` 是否还能保持在接近 `v3` 的水平
- `two_stage_gate_score` 的 SAT/DSAT 分布是否更接近真实分布
