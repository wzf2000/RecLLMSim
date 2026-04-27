# Memory V3.1 Design

## Goal

`memory_v3` 的首轮结果表明：

- calibration 分离思路是有效的
- 但当前 `v3` 把“证据不足时保守”解释得过头，导致模型明显偏 SAT

`v3.1` 的目标不是改 memory cache，而是只改 turn evaluation prompt：

- 保留 `memory_v3` 的 calibration 优势
- 把 `3/4` 最低满意线重新变成强 gate
- 明确规定：证据不足只影响 `1/2/3` 内部细分，不影响先把样本判成 `<=3`

## Main Change

相比 `v3`，`v3.1` 的关键变化是：

1. `Step B` 变成强 `3/4` gate  
   必须先判断：
   - 核心问题是否回答
   - 关键约束 / 关键任务目标 / 用户特定要求是否满足
   - 剩余问题是否只是普通缺口

2. calibration 只作为整体刻度先验  
   不再允许它直接放松 SAT/DSAT 判断

3. `3→4` 边界即使是弱推断，也不能因此默认偏 SAT  
   弱推断只意味着：
   - 少依赖文本边界描述
   - 多依赖核心问题、关键约束、真实案例

4. 低分证据 sparse/none 只影响 `1/2/3`  
   若未通过 SAT gate，仍然应该先落到 `<=3`
   只是内部细分时默认优先给 `3`

## Recommended Run

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
memory_version=v3 \
turn_eval_prompt_version=v3_1 \
limit_users=20 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_1_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

## What to Watch

这版最关键的观察点：

- `MAE / Pearson / Spearman` 能否接近或保持 `v3`
- `F1-DSAT` 是否从 `v3` 的低位明显回升
- `false_sat_rate` 是否显著下降
- 预测分布是否不再严重塌到 `4/5`
