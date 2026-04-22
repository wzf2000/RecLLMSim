# Boundary 3/4 Selective Refute Prompt Design

## 1. Background

此前两个 refute 版本暴露出相反的问题：

- `boundary_34_refute`
  - 明显提高了对不满意样本的抓取能力
  - 但过度保守，预测分布明显偏向 `3`
- `boundary_34_refute_v2`
  - 为了缓和保守倾向，放宽了判 `3` 的条件
  - 但在 10 用户子集上又几乎退化成“默认判满意”

这说明问题不在于“要不要做 refute”，而在于：

1. 不应该对所有样本都做同样强度的 refute
2. refute 更适合用于真正接近 `3/4` 边界的样本

## 2. New Design

新增 prompt 版本：`boundary_34_selective_refute`

核心思路：

- 第一遍只做温和的 `3/4` 初判
- 同时要求模型判断：这个样本是否真的接近 `3/4` 边界
- 只有第一遍显式输出 `needs_refute_review=true` 时，才进入第二遍复核
- 第二遍不再重新完整评分，只检查第一遍指出的“可疑点”是否真的足以跨过满意边界

## 3. Two-Pass Flow

### 3.1 First Pass: Mild Boundary Decision

第一遍 prompt 的目标：

- 输出 `classification ∈ {3, 4}`
- 输出 `reason`
- 输出简短 `analysis`
- 额外输出 `needs_refute_review: true/false`

第一遍的判定原则：

- 先温和判断是否达到满意最低线
- 只有在证据混合、判断明显接近 `3/4` 边界时，才允许触发复核
- 明显满意或明显不满意都不触发复核

`needs_refute_review=true` 的典型情形：

- 回复大体有帮助，但可能漏掉了一个关键要求
- 当前判成 `3`，但主要问题也可能只是“不够细致”
- 当前判成 `4`，但是否真的跨过满意线仍不稳

### 3.2 Second Pass: Short Refute Follow-Up

第二遍只在第一遍触发时运行。

第二遍 prompt 的目标：

- 不重新展开完整评分流程
- 只围绕第一遍指出的“可疑点”做短复核
- 判断这个可疑点是否真的是关键失败

第二遍的核心规则：

- 如果只是普通缺口、轻度不够细致、但不影响核心可用性，应保护 `4`
- 只有当该问题确实导致核心问题未被回答、关键要求被忽略、或明显低于最低满意线时，才判 `3`

## 4. Implementation Details

代码实现位置：

- `detection/lib/memory.py`
  - 新增 `boundary_34_selective_refute` 第一遍 prompt
  - 新增 `build_turn_eval_refute_followup_prompt()` 第二遍 prompt
- `detection/trace/collect_personalized.py`
  - 新增 `SelectiveBoundaryTurnPrediction`
  - 新增 `_predict_turn_with_optional_selective_refute()`
  - 第一遍输出 `needs_refute_review=true` 时，自动触发第二遍复核
  - 输出结果中附加：
    - `analysis_first_pass`
    - `analysis_refute`
    - `selective_refute_triggered`
    - `selective_refute_applied`
    - `selective_refute_initial_score`
    - `selective_refute_initial_reason`

路由策略：

- 第一遍 `boundary_34_selective_refute` 走 boundary raw-text parse 路线
- 第二遍内部使用 `boundary_34_selective_refute_followup`
- follow-up parse 失败时，不中断整条样本，直接回退到第一遍结果

## 5. Expected Behavior

预期该设计相对前两个 refute 版本有两个好处：

1. 避免对明显满意样本施加过强的 refute 偏置
2. 把额外计算集中在真正接近 `3/4` 边界的样本上

理想结果不是简单追求更高的 DSAT 召回，而是：

- 比 `boundary_34_refute_v2` 更能识别不满意
- 比 `boundary_34_refute` 更少误伤真实满意
- 在 `F1-DSAT / false_sat_rate / false_dsat_rate` 之间取得更平衡的折中

## 6. Suggested Evaluation Order

建议先跑用户子集而不是直接全量：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_selective_refute \
limit_users=10 \
bash scripts/collect_personalized_vllm.sh
```

若 10 用户子集上出现下面任一情况，再考虑全量：

- `selective_refute_triggered` 比例明显低于全量 turns，说明它确实只在边界样本上触发
- `F1-DSAT` 不低于 `boundary_34_refute`
- `false_dsat_rate` 明显低于 `boundary_34_refute`
- 预测满意/不满意比例不再像 `refute_v1` 或 `refute_v2` 那样严重偏斜
