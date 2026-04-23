# Boundary 3/4 Selective Refute V2 Reason-Fix Anchor2 Full Results

## 1. Setup and Comparison Boundary

本次新结果文件：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_reasonfix_anchor2.jsonl`

规模：

- `90` 个用户
- `6474` 个 assistant turns

当前全量可直接对比的旧文件是：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2.jsonl`

**重要说明：这不是一个纯 anchor ablation。**

旧文件和新文件同时存在两类差异：

1. 新文件加入了 `reason-fix`
2. 新文件使用了 `n_anchors=2`

因此下面的全量比较只能回答：

- `reason-fix + anchor2` 这个新系统，相比旧版全量 `v2`，整体表现如何

它**不能严格隔离 anchor2 本身的净贡献**。

## 2. Sanity Check

旧版全量 `v2`：

- 非法 `score/reason` 组合：`3281`
  - 全部是 `pred_score >= 4` 但 `reason != 满意`

新全量 `reasonfix + anchor2`：

- 非法 `score/reason` 组合：`0`

所以从语义一致性上看，新文件显著更干净。

## 3. Main Metric Comparison

| Metric | old full v2 | reasonfix+anchor2 | delta |
|---|---:|---:|---:|
| MAE | `0.8015` | `0.8023` | `+0.0008` |
| Pearson | `0.1923` | `0.1793` | `-0.0130` |
| Spearman | `0.1688` | `0.1504` | `-0.0184` |
| QWK | `0.1229` | `0.1147` | `-0.0082` |
| Boundary Acc | `0.7260` | `0.7252` | `-0.0008` |
| F1-DSAT | `0.3177` | `0.3197` | `+0.0020` |
| Boundary Kappa | `0.1510` | `0.1526` | `+0.0016` |
| AUC | `0.5859` | `0.5873` | `+0.0014` |
| False SAT | `0.6269` | `0.6224` | `-0.0045` |
| False DSAT | `0.2012` | `0.2031` | `+0.0019` |

## 4. User-Aware Comparison

### 4.1 Global user-aware metrics

| Metric | old full v2 | reasonfix+anchor2 | delta |
|---|---:|---:|---:|
| PU-Pearson | `0.1602` | `0.1490` | `-0.0112` |
| PU-Spearman | `0.1552` | `0.1398` | `-0.0154` |
| PU-Kappa | `0.0910` | `0.0850` | `-0.0060` |
| WC-Pearson | `0.1721` | `0.1620` | `-0.0101` |

### 4.2 User-aware binary metrics

| Metric | old full v2 | reasonfix+anchor2 | delta |
|---|---:|---:|---:|
| PU-bin F1-DSAT | `0.2765` | `0.2736` | `-0.0029` |
| PU-bin Kappa | `0.1175` | `0.1190` | `+0.0015` |
| PU-bin AUC | `0.5940` | `0.5792` | `-0.0148` |
| WC-bin Pearson | `0.1323` | `0.1377` | `+0.0054` |

## 5. Distribution and Trigger Diagnostics

### 5.1 Prediction distribution

旧版全量 `v2`：

- pred: `SAT=4981`, `DSAT=1493`

新全量 `reasonfix + anchor2`：

- pred: `SAT=4966`, `DSAT=1508`

变化非常小，整体仍然是一个轻微偏向 `DSAT` 的边界系统。

### 5.2 Selective trigger

旧版全量 `v2`：

- `selective_refute_model_flag = 914`
- `selective_refute_triggered = 679`

新全量 `reasonfix + anchor2`：

- `selective_refute_model_flag = 1080`
- `selective_refute_triggered = 961`

因此：

- 触发量明显增加
- second-pass 在全量上比旧版更活跃

### 5.3 Second-pass transition pattern

新全量 `reasonfix + anchor2` 的主要 transition：

- `(4 -> 4)`: `4953`
- `(3 -> 3)`: `1500`
- `(3 -> 4)`: `13`
- `(4 -> 3)`: `8`

相比旧版：

- 旧版几乎只有 `3 -> 4`
- 新版开始出现少量 `4 -> 3`

这说明：

- 新系统的 second-pass 已不再只是“保守地往 SAT 拉回”
- 但改判总量依然很小，整体主导因素仍然是 first-pass

## 6. Interpretation

当前最合理的结论是：

**`reasonfix + anchor2` 在全量上没有形成明确提升，更像是“语义修正后整体基本持平，边界 DSAT 指标略好一点，但全局和用户级相关性略差一点”。**

更具体地说：

- 好处：
  - reason 语义完全正确
  - `F1-DSAT / false_sat_rate / boundary kappa / AUC` 有轻微改善
- 代价：
  - `Pearson / Spearman / QWK` 回落
  - 多数 user-aware 指标略降
  - second-pass 触发更多，但没有换来明显增益

因此，这版更像：

- 一个语义上更干净的系统
- 但从纯性能角度看，没有明显超越旧版全量 `v2`

## 7. Practical Conclusion

如果问题是：

- `anchor=2` 值不值得继续往下加？

我的判断是：

- **不值得再继续往更大 anchor 数扫了**

理由：

- 在 20 用户子集上，`anchor1` 有小幅正向信号
- 到全量 `anchor2`，这个信号没有稳定放大
- 反而更像回到“整体几乎持平，局部小幅摇摆”

因此后续更值得继续投入的方向仍然是：

- first-pass 本身的边界判断语言

而不是继续扩展 anchor 数量。
