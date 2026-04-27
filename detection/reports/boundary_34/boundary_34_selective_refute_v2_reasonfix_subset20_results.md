# Boundary 3/4 Selective Refute V2 Reason-Fix Subset20 Results

## 1. Setup

本次分析比较的是：

- 旧版 `boundary_34_selective_refute_v2` 20 用户子集结果
- 加入 `reason/score` 合法性规则后的新结果

新结果文件：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_reasonfix_u20.jsonl`

旧版对照基线来自已有报告：

- `detection/reports/boundary_34/boundary_34_selective_refute_v2_subset_results.md`

两者都基于：

- `20` 个用户
- `1594` 个 assistant turns

## 2. Reason Legality Check

新结果中，`reason` 与 `pred_score` 的组合已完全合法：

- `pred_score >= 4` 且 `reason != 满意`: `0`
- `pred_score <= 3` 且 `reason == 满意`: `0`
- 非法组合总数: `0`

这说明本次修改在工程层面已生效：

- prompt 显式写入了合法性规则
- 采集后处理也会对非法输出做归一化

## 3. Main Metric Comparison

| Metric | old v2 | reason-fix v2 | delta |
|---|---:|---:|---:|
| MAE | `0.7955` | `0.7967` | `+0.0012` |
| Pearson | `0.2228` | `0.2215` | `-0.0013` |
| QWK | `0.1442` | `0.1436` | `-0.0006` |
| Boundary Acc | `0.7315` | `0.7302` | `-0.0013` |
| F1-DSAT | `0.3495` | `0.3524` | `+0.0029` |
| False SAT | `0.6048` | `0.5979` | `-0.0069` |
| False DSAT | `0.1934` | `0.1965` | `+0.0031` |
| PU-bin F1-DSAT | `0.3067` | `0.3027` | `-0.0040` |
| PU-bin Kappa | `0.1474` | `0.1449` | `-0.0025` |
| WC-bin Pearson | `0.1626` | `0.1705` | `+0.0079` |

## 4. Distribution Shift

真实分布不变：

- gold: `SAT=1303`, `DSAT=291`

预测分布从：

- old v2: `SAT=1227`, `DSAT=367`

变成：

- reason-fix v2: `SAT=1221`, `DSAT=373`

这说明 reason-fix 后模型整体略微更偏向 `DSAT`，但幅度很小。

## 5. Selective Diagnostics

### 5.1 Trigger rate increased

旧版 v2：

- `selective_refute_model_flag = 256`
- `selective_refute_triggered = 185`
- 触发率：`11.6%`

reason-fix v2：

- `selective_refute_model_flag = 297`
- `selective_refute_triggered = 241`
- 触发率：`15.1%`

这是本次修改最明显的系统层变化。

原因也比较清楚：

- 旧版里 `classification=4` 的触发 gate 依赖“不满意 reason”
- 修正后 `classification=4` 的 `reason` 必须为 `满意`
- 因此代码侧改成：对 `4` 分样本只要 first-pass 给出 `needs_refute_review=true` 就允许触发

### 5.2 Trigger composition

新结果触发样本构成：

- `initial_score=3`, `initial_reason=不够细致`: `237`
- `initial_score=4`, `initial_reason=满意`: `4`

也就是说：

- 绝大多数触发仍然来自旧主模式：`3 + 不够细致`
- 但现在开始出现少量 `4 + 满意` 的复核样本

### 5.3 Second pass became slightly more active

旧版 v2：

- 触发 `185`
- 真正改判 `2`
- change rate: `1.1%`

reason-fix v2：

- 触发 `241`
- 真正改判 `6`
- change rate: `2.5%`

改判模式：

- `(3 -> 3)`: `231`
- `(3 -> 4)`: `6`
- `(4 -> 4)`: `4`

其中：

- `3 -> 4` 修正了真实 SAT：`5`
- `3 -> 4` 误伤真实 DSAT：`1`
- `4 -> 3`: `0`

这说明 reason-fix 后第二遍依然总体偏保守，但比旧版稍微更“活”了一点。

## 6. Interpretation

这次 reason-fix 的结论可以概括成一句：

**它主要修正了语义一致性，没有显著改变 v2 的边界性能画像。**

更具体地说：

- 工程目标已经达成：`reason` 现在与 `score` 完全一致
- 边界性能没有明显崩，也没有明显跃升
- `F1-DSAT` 和 `false_sat_rate` 有轻微改善
- 但 `MAE / Pearson / PU-bin F1-DSAT / PU-bin Kappa` 有轻微回落
- 整体幅度都很小，属于“系统语义修正带来的小范围分布漂移”

所以目前最合理的判断是：

- `boundary_34_selective_refute_v2` 仍然是当前 selective 主线版本
- reason-fix 应该视为**必要的语义修复**，不是一次以提分为目标的 prompt 优化

## 7. Recommendation

后续如果继续沿 v2 往下做，应把这版 reason-fix 作为新的默认基线。

原因：

- 语义定义终于和 ground truth 对齐
- 指标没有明显恶化
- 后面再做 first-pass 优化时，reason 字段的解释会更可靠
