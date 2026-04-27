# Boundary 3/4 Selective Refute Results on 20-User Subset

## 1. Setup

本次分析对象：

- 方法：`boundary_34_selective_refute`
- backbone：`Qwen/Qwen3-8B`
- memory update：`none`
- 数据文件：`detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute.jsonl`

子集规模：

- `20` 个用户
- `1594` 个 assistant turns
- 真实 SAT / DSAT 分布：
  - SAT (`>=4`) = `1303` (`81.7%`)
  - DSAT (`<=3`) = `291` (`18.3%`)

为了避免“子集 vs 全量”混比，下面所有对照方法都只在这 `1594` 个相同 `sample_id` 上重新计算：

- `Qwen none`
- `boundary_34`
- `boundary_34_refute`
- `boundary_34_selective_refute`

说明：

- `boundary_34_refute_v2` 当前只有 10 用户结果，不纳入本轮 20 用户对照表。

## 2. Main Results

### 2.1 Same-Subset Comparison

| Method | MAE | Pearson | QWK | Acc | F1-DSAT | False SAT | False DSAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7077` | `0.2999` | `0.2807` | `0.7629` | `0.3505` | `0.6495` | `0.1450` | `0.2998` | `0.1520` | `0.1777` |
| `boundary_34` | `0.7302` | `0.1497` | `0.0749` | `0.7967` | `0.2322` | `0.8316` | `0.0629` | `0.1875` | `0.1053` | `0.1512` |
| `boundary_34_refute` | `0.8369` | `0.2325` | `0.1525` | `0.6901` | `0.3683` | `0.5052` | `0.2663` | `0.3317` | `0.1544` | `0.1718` |
| `boundary_34_selective_refute` | `0.7992` | `0.1959` | `0.1265` | `0.7277` | `0.3323` | `0.6289` | `0.1926` | `0.2862` | `0.1264` | `0.1459` |

### 2.2 Distribution of SAT / DSAT Predictions

| Method | Pred SAT | Pred DSAT | SAT Ratio | DSAT Ratio |
|---|---:|---:|---:|---:|
| Gold | `1303` | `291` | `81.7%` | `18.3%` |
| `Qwen none` | `1303` | `291` | `81.7%` | `18.3%` |
| `boundary_34` | `1463` | `131` | `91.8%` | `8.2%` |
| `boundary_34_refute` | `1103` | `491` | `69.2%` | `30.8%` |
| `boundary_34_selective_refute` | `1235` | `359` | `77.5%` | `22.5%` |

## 3. Key Findings

### 3.1 Selective version is more balanced than `refute_v1`, but still not better than `Qwen none`

`boundary_34_selective_refute` 的位置介于两个极端之间：

- 相比 `boundary_34`
  - 更能抓不满意：
    - `F1-DSAT: 0.2322 -> 0.3323`
    - `false_sat_rate: 0.8316 -> 0.6289`
- 相比 `boundary_34_refute`
  - 更少误伤满意：
    - `false_dsat_rate: 0.2663 -> 0.1926`
    - `Accuracy: 0.6901 -> 0.7277`

但它仍然没有超过当前最稳的 raw baseline `Qwen none`：

- `F1-DSAT: 0.3505 -> 0.3323`（下降）
- `false_sat_rate: 0.6495 -> 0.6289`（略有改善）
- `false_dsat_rate: 0.1450 -> 0.1926`（明显变差）
- `PU-bin F1-DSAT: 0.2998 -> 0.2862`（下降）
- `PU-bin Kappa: 0.1520 -> 0.1264`（下降）
- `WC-bin Pearson: 0.1777 -> 0.1459`（下降）

结论：

- 这版没有像 `refute_v1` 那样明显过保守
- 但也还没有达到“在个性化满意边界上真正优于 `Qwen none`”的程度

### 3.2 It successfully moved away from both previous extremes

这版最清晰的正面信号不是“绝对最优”，而是：

- 不再像 `boundary_34` 一样严重偏 SAT
- 也不再像 `boundary_34_refute` 一样明显偏 DSAT

预测分布从：

- `boundary_34`: `91.8 / 8.2`
- `boundary_34_refute`: `69.2 / 30.8`

拉回到：

- `boundary_34_selective_refute`: `77.5 / 22.5`

这比前两个边界 prompt 都更接近真实分布 `81.7 / 18.3`。

### 3.3 The personalization signal is still weaker than the raw baseline

虽然 selective 版本在分布上更平衡，但个性化指标没有相应提升：

- `PU-Pearson: 0.2007 -> 0.1563`
- `WC-Pearson: 0.2154 -> 0.1672`
- `PU-bin F1-DSAT: 0.2998 -> 0.2862`
- `PU-bin Kappa: 0.1520 -> 0.1264`

这说明当前两阶段机制仍然更像是在做全局边界校准，
而不是更精确地提升“每个用户自己的满意阈值判断”。

## 4. Selective-Refute Diagnostics

### 4.1 Trigger rate is high

在 `1594` 个样本中：

- `selective_refute_triggered = 764`
- 触发率 = `47.9%`

这比“只对少量边界样本触发”要高得多。  
说明第一遍 prompt 仍然把太多样本判成“接近边界”，selective 还不够 selective。

### 4.2 Most triggered samples do not change after follow-up

- 触发复核的样本：`764`
- 最终真正改判的样本：`142`
- `change rate | triggered = 18.6%`

初判到终判的变化：

- `(4 -> 4)`: `311`
- `(3 -> 3)`: `311`
- `(3 -> 4)`: `113`
- `(4 -> 3)`: `29`

也就是说：

- 第二遍大多数时候只是“确认第一遍”
- 真正发生的改判以 `3 -> 4` 为主

### 4.3 Follow-up mainly protects SAT, but not enough to beat baseline

这 `142` 个改判里：

- `3 -> 4` 且对真实 SAT 有利：`80`
- `3 -> 4` 但伤害真实 DSAT：`33`
- `4 -> 3` 且对真实 DSAT 有利：`4`
- `4 -> 3` 但误伤真实 SAT：`25`

这说明第二遍的主要作用是：

- 把一部分“第一遍过保守”的 SAT 样本拉回 `4`

但它的副作用仍然存在：

- 也会把一些真实 DSAT 拉成 SAT
- 且少量 `4 -> 3` 改判误伤了真实 SAT

### 4.4 First-pass alone is actually better at DSAT capture

如果只看第一遍初判（不做 follow-up），它的边界指标是：

- `F1-DSAT = 0.3733`
- `false_sat_rate = 0.5292`
- `false_dsat_rate = 0.2348`

而最终两阶段结果变成：

- `F1-DSAT = 0.3323`
- `false_sat_rate = 0.6289`
- `false_dsat_rate = 0.1926`

这说明第二遍 follow-up 的主要作用是：

- 降低过保守倾向
- 但同时削弱了 DSAT 的抓取能力

换句话说，当前第二遍 prompt 仍然偏向“保护 SAT”，力度有点过强。

## 5. Task-Level Notes

当前 selective 版本按任务的边界表现：

| Task | Accuracy | F1-DSAT |
|---|---:|---:|
| `旅行规划` | `0.7451` | `0.3723` |
| `技能学习规划` | `0.7493` | `0.2500` |
| `礼物准备` | `0.6674` | `0.3260` |
| `菜谱规划` | `0.7649` | `0.3577` |

可以看到：

- `旅行规划 / 菜谱规划` 上的 DSAT 抓取相对更好
- `技能学习规划` 上的 `F1-DSAT` 仍然最低
- `礼物准备` 依然是整体最难的任务之一（Accuracy 最低）

## 6. Conclusion

当前 `boundary_34_selective_refute` 的结论是：

1. 它确实比前两个 boundary prompt 更接近一个“中间态”
2. 但还没有超过 `Qwen none` 这个 raw baseline
3. 第一遍已经能抓到较多 DSAT，问题主要出在第二遍 follow-up 过度偏向保护 SAT
4. selective 触发率仍然太高，说明第一遍的“是否需要复核”判据还不够严格

因此，这版适合作为继续迭代的方向，但**不建议直接全量替代主线方法**。

## 7. Next Design Implications

下一步最值得改的不是第一遍，而是第二遍 follow-up：

- 收紧 `needs_refute_review=true` 的触发条件，让触发率明显低于当前 `47.9%`
- 第二遍不要默认保护 `4`
- 第二遍应明确区分：
  - “普通不够细致，但仍可用”
  - “关键缺失，确实低于满意线”
- 第二遍最好更多服务于“核实第一遍的可疑点是否真的是关键失败”，而不是泛化地往 SAT 回摆

如果继续沿这条路线，下一版更像应该是：

- `selective_refute_v2`
  - 更少触发
  - 更弱的 SAT 保护
  - 更强的“关键失败核实”约束
