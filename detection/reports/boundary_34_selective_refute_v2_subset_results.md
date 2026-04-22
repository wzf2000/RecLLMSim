# Boundary 3/4 Selective Refute V2 Subset Results

## 1. Important Note About This Run

用户反馈为“已经跑完了 40 个用户”，但当前结果文件：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2.jsonl`

实际只包含：

- `20` 个用户
- `1594` 个 assistant turns
- `346` 个 block

因此，本报告分析的是**当前文件里真实存在的 20 用户子集结果**，而不是 40 用户结果。

这通常说明：

- 续跑时沿用了旧输出文件
- 或者当前输出文件仍然对应之前的 20 用户子集

如果后续要得到真正的 40 用户结果，建议显式指定新的 `output_jsonl` 文件名，或者先清理旧文件再跑。

## 2. Evaluation Summary

当前文件评测结果：

- `MAE = 0.7955`
- `Pearson = 0.2228`
- `QWK = 0.1442`
- `Boundary Accuracy = 0.7315`
- `F1-DSAT = 0.3495`
- `false_sat_rate = 0.6048`
- `false_dsat_rate = 0.1934`

用户感知二分类指标：

- `PU-bin F1-DSAT = 0.3067`
- `PU-bin Kappa = 0.1474`
- `WC-bin Pearson = 0.1626`

预测分布：

- gold: `SAT=1303`, `DSAT=291`
- pred: `SAT=1227`, `DSAT=367`

## 3. Same-Subset Comparison

下面所有方法都在**同一批 1594 个 sample_id**上重算：

| Method | MAE | Pearson | QWK | Acc | F1-DSAT | False SAT | False DSAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7077` | `0.2999` | `0.2807` | `0.7629` | `0.3505` | `0.6495` | `0.1450` | `0.2998` | `0.1520` | `0.1777` |
| `boundary_34` | `0.7302` | `0.1497` | `0.0749` | `0.7967` | `0.2322` | `0.8316` | `0.0629` | `0.1875` | `0.1053` | `0.1512` |
| `boundary_34_refute` | `0.8369` | `0.2325` | `0.1525` | `0.6901` | `0.3683` | `0.5052` | `0.2663` | `0.3317` | `0.1544` | `0.1718` |
| `boundary_34_selective_refute` | `0.7992` | `0.1959` | `0.1265` | `0.7277` | `0.3323` | `0.6289` | `0.1926` | `0.2862` | `0.1264` | `0.1459` |
| `boundary_34_selective_refute_v2` | `0.7955` | `0.2228` | `0.1442` | `0.7315` | `0.3495` | `0.6048` | `0.1934` | `0.3067` | `0.1474` | `0.1626` |

## 4. Main Conclusions

### 4.1 V2 clearly improves over selective v1

相对 `boundary_34_selective_refute`：

- `MAE: 0.7992 -> 0.7955`
- `Pearson: 0.1959 -> 0.2228`
- `QWK: 0.1265 -> 0.1442`
- `Acc: 0.7277 -> 0.7315`
- `F1-DSAT: 0.3323 -> 0.3495`
- `false_sat_rate: 0.6289 -> 0.6048`
- `PU-bin F1-DSAT: 0.2862 -> 0.3067`
- `PU-bin Kappa: 0.1264 -> 0.1474`
- `WC-bin Pearson: 0.1459 -> 0.1626`

也就是说，v2 的收紧触发 + 更保守的第二遍复核是有效的。

### 4.2 V2 is now very close to `Qwen none` on boundary performance

和当前最稳的 raw baseline `Qwen none` 相比：

- `F1-DSAT: 0.3505 vs 0.3495`
- `PU-bin Kappa: 0.1520 vs 0.1474`
- `PU-bin F1-DSAT: 0.2998 vs 0.3067`

这说明：

- V2 在整体边界能力上已经基本追平 `Qwen none`
- 在 `PU-bin F1-DSAT` 上甚至略高于 `Qwen none`

但它仍然没有全面超过 `Qwen none`：

- `MAE / Pearson / QWK` 仍弱于 `Qwen none`
- `false_dsat_rate = 0.1934` 仍高于 `Qwen none = 0.1450`

所以当前最准确的判断是：

- `boundary_34_selective_refute_v2` 已经从“明显不如 baseline”进展到“接近 baseline，部分个性化边界指标可比甚至略优”
- 但还不能说已经稳定优于 `Qwen none`

### 4.3 V2 is substantially better than `boundary_34` and more balanced than `refute_v1`

相对两个更早的边界 prompt：

- 比 `boundary_34` 强很多
  - `F1-DSAT: 0.2322 -> 0.3495`
  - `false_sat_rate: 0.8316 -> 0.6048`
- 比 `boundary_34_refute` 更平衡
  - `Acc: 0.6901 -> 0.7315`
  - `false_dsat_rate: 0.2663 -> 0.1934`
  - `F1-DSAT` 仅略低于 `refute_v1` (`0.3683 -> 0.3495`)

这说明 v2 基本达到了本轮设计目标：

- 保留一部分 DSAT 抓取能力
- 但不再像 `refute_v1` 那样过度保守

## 5. Selective Diagnostics

### 5.1 Trigger condition is much tighter now

在 v2 中：

- `selective_refute_model_flag = 256`
- `selective_refute_triggered = 185`

对应比例：

- 模型建议复核率：`16.1%`
- 实际触发率：`11.6%`

对比 v1：

- v1 实际触发率：`47.9%`
- v2 实际触发率：`11.6%`

这说明两层收紧都起作用了：

1. 第一遍 prompt 本身触发更少
2. 代码 gate 进一步拦掉了一部分不必要复核

### 5.2 Second pass now almost never changes the prediction

在这 `185` 个触发样本里：

- 真正改判的只有 `2` 个
- `change rate | triggered = 1.1%`

改判模式：

- `(3 -> 3)`: `183`
- `(3 -> 4)`: `2`

而且这两个 `3 -> 4` 都是对真实 SAT 的正向修正，没有引入新的错误：

- `3 -> 4` help SAT: `2`
- `3 -> 4` hurt DSAT: `0`
- `4 -> 3` help DSAT: `0`
- `4 -> 3` hurt SAT: `0`

这说明 v2 的第二遍现在已经从“主动拉回 SAT”变成了：

- 几乎总是维持第一遍
- 极少数情况下做安全的 `3 -> 4` 修正

### 5.3 Final result is now almost identical to first pass

first-pass 指标：

- `F1-DSAT = 0.3485`
- `false_sat_rate = 0.6048`
- `false_dsat_rate = 0.1949`

final 指标：

- `F1-DSAT = 0.3495`
- `false_sat_rate = 0.6048`
- `false_dsat_rate = 0.1934`

这说明当前 v2 的第二遍基本不再主导结果，整体性能几乎完全由第一遍决定。

这既是优点，也是限制：

- 优点：不会再像 v1 那样把第二遍变成大规模偏 SAT 的后处理
- 限制：当前第二遍的实际贡献已经很小

## 6. Trigger Pattern

当前触发样本的初判原因完全塌缩到：

- `不够细致`: `185`

这说明代码 gate 虽然把触发量收紧了，但第二遍目前几乎只在处理：

- “第一遍判 3，理由是 `不够细致`” 这一类样本

从任务分布看，触发主要集中在：

- `礼物准备`: `70`
- `旅行规划`: `60`
- `菜谱规划`: `28`
- `技能学习规划`: `27`

## 7. Interpretation

当前 `boundary_34_selective_refute_v2` 的本质是：

- 第一遍已经变成一个相对稳的边界分类器
- 第二遍不再是一个强干预模块，而更像一个很保守的 safety check

因此，这一版的真正收益主要来自：

- 更严格的第一遍触发语义
- 代码层的 trigger gate

而不是第二遍 prompt 本身带来了大量有效修正。

## 8. Next Step Recommendation

如果继续沿 selective 这条路线走，下一步重点不应该再放在“继续收紧第二遍”，因为它已经几乎不动了。

更值得做的是两种方向二选一：

### Option A: Keep the current v2 and stop here

理由：

- 它已经比 v1 明显更好
- 已经接近 `Qwen none`
- 复杂度和风险都比继续放大第二遍更可控

### Option B: Make the first pass better instead of making the second pass stronger

因为当前结果几乎由 first-pass 决定，所以如果想继续提升：

- 应该改第一遍 prompt
- 尤其是让第一遍更精确地区分：
  - “只是普通的不够细致”
  - “虽然看起来在帮忙，但实际上没过满意线”

一个具体方向是：

- 保留 v2 的触发机制不动
- 直接优化 first-pass 的 `3/4` 边界判断语言
- 把第二遍继续当作极少触发的保险机制

## 9. Practical Note

如果你确实想跑 40 用户结果，建议显式使用新文件名，例如：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_selective_refute_v2 \
limit_users=40 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_u40.jsonl \
bash scripts/collect_personalized_vllm.sh
```

否则当前默认文件会让后续分析看起来像是“40 用户”，但实际上仍然只分析到了旧的 20 用户子集。
