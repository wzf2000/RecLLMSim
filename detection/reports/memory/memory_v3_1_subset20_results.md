# Memory V3.1 Subset20 Results

## 1. Setup

本次分析文件：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_1_u20.jsonl`

规模：

- `20` 个用户
- `1594` 个 assistant turns

同子集对照方法：

- `memory_v3`
- `Qwen none`
- `boundary_34_selective_refute_v2`
- `Qwen no_memory`

注意：以下所有主对比都在**同一批 1594 个 sample_id**上重算。

## 2. Main Comparison

| Method | MAE | Pearson | Spearman | QWK | Acc | F1-DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `memv3_1` | `0.6418` | `0.2942` | `0.2997` | `0.2475` | `0.8043` | `0.1789` | `0.8832` | `0.0422` |
| `memv3` | `0.6349` | `0.3163` | `0.3176` | `0.2687` | `0.8168` | `0.1705` | `0.8969` | `0.0238` |
| `Qwen none` | `0.7077` | `0.2999` | `0.3012` | `0.2807` | `0.7629` | `0.3505` | `0.6495` | `0.1450` |
| `boundary_v2` | `0.7955` | `0.2228` | `0.2123` | `0.1442` | `0.7315` | `0.3495` | `0.6048` | `0.1934` |
| `Qwen no_memory` | `0.7917` | `0.1392` | `0.1069` | `0.0706` | `0.8174` | `0.0459` | `0.9759` | `0.0054` |

## 3. What Improved Relative to V3

`v3.1` 相对 `v3` 的改善是存在的，但幅度很小：

- `F1-DSAT: 0.1705 -> 0.1789`
- `false_sat_rate: 0.8969 -> 0.8832`
- `pred_score=3`: `55 -> 85`
- `pred_score=5`: `636 -> 393`

这说明：

- 强化 `3/4` gate 确实开始起作用
- 模型不再像 `v3` 那样那么容易直接给 `5`
- 也更愿意把一部分样本拉回 `3/4`

所以方向上，`v3.1` 是**修正了 `v3` 的一部分 SAT 偏置**。

## 4. What Got Worse

但 `v3.1` 的代价也很直接：

- `MAE: 0.6349 -> 0.6418`
- `Pearson: 0.3163 -> 0.2942`
- `Spearman: 0.3176 -> 0.2997`
- `QWK: 0.2687 -> 0.2475`
- `Accuracy: 0.8168 -> 0.8043`
- `AUC: 0.6574 -> 0.6229`

所以 `v3.1` 的画像是：

- **比 `v3` 更不那么离谱地偏 SAT**
- 但为了换回一点 DSAT，损失了不少 `v3` 的 calibration 优势

## 5. Distribution Diagnosis

真实分布：

- `1: 38`
- `2: 72`
- `3: 181`
- `4: 611`
- `5: 692`

`v3.1` 预测分布：

- `2: 4`
- `3: 85`
- `4: 1112`
- `5: 393`

对比 `v3`：

- `v3`: `{2: 6, 3: 55, 4: 897, 5: 636}`
- `v3.1`: `{2: 4, 3: 85, 4: 1112, 5: 393}`

这意味着：

- `v3.1` 主要做的是把一部分 `5` 拉回 `4`
- 也增加了一些 `3`
- 但仍然**几乎不给真正的低分**

尤其：

- 没有任何 `1` 分预测
- `2/3` 总量仍远低于真实低分分布

所以当前的修正更多是：

- `5 -> 4`
- 而不是：
  - `4/5 -> 3`
  - `3 -> 2/1`

## 6. Boundary Interpretation

SAT / DSAT：

- gold: `SAT=1303`, `DSAT=291`
- `v3.1`: `SAT=1505`, `DSAT=89`
- `v3`: `SAT=1533`, `DSAT=61`

可见：

- `v3.1` 还是显著偏 SAT
- 只是偏得比 `v3` 略轻

和 `Qwen none` 对比：

- `Qwen none` 的 `F1-DSAT = 0.3505`
- `v3.1` 的 `F1-DSAT = 0.1789`

所以 `v3.1` 还远没有回到一个“合理的 DSAT 边界系统”。

## 7. User-Aware Impact

相对 `v3`，用户级指标也是回落的：

- `PU-Pearson: 0.1996 -> 0.1745`
- `PU-Kappa: 0.1554 -> 0.1362`
- `PU-bin F1-DSAT: 0.1425 -> 0.1463`（仅极小提升）
- `PU-bin Kappa: 0.0910 -> 0.0759`
- `WC-bin Pearson: 0.1629 -> 0.1258`

因此 `v3.1` 没有形成“用户级个性化更好”的收益，它更像一个：

- 用少量 DSAT 回补，换掉部分 `v3` 的用户级 calibration 表现

## 8. Memory Usage Signal

block-level 相关性：

- `v3`: `corr(memory_avg, pred_block_mean) = 0.6305`
- `v3.1`: `corr(memory_avg, pred_block_mean) = 0.6638`

这很重要。

说明：

- `v3.1` 并没有减少对 memory calibration 的使用
- 反而更强地跟随了 memory 的均值刻度

严格度分桶也能看出同样趋势：

### strict

- gold mean = `3.676`
- `v3` pred mean = `4.124`
- `v3.1` pred mean = `3.971`

### mid

- gold mean = `3.923`
- `v3` pred mean = `4.226`
- `v3.1` pred mean = `4.090`

### lenient

- gold mean = `4.550`
- `v3` pred mean = `4.563`
- `v3.1` pred mean = `4.356`

解释：

- `v3.1` 确实把 strict / mid 桶往下拉了一点
- 但整体仍明显偏 SAT
- 并且在 lenient 桶上开始出现轻微往下压

所以它像是一个“整体回缩”的修正，而不是一个真正把 `3/4` gate 做准的修正。

## 9. Bottom Line

`memory_v3.1` 的结论可以压成一句话：

**它确实把 `v3` 的 SAT 偏置往回拉了一点，但主要是通过把一部分 `5` 压回 `4` 完成的，而不是显著恢复了对 DSAT 的识别能力。**

因此：

- 它方向上优于 `v3`
- 但还不够好，不能作为新的稳定版本

## 10. Practical Next Step

如果继续沿 `v3` 系列走，我不建议再继续只靠文字上“提醒不要默认 SAT”。

更值得做的是：

1. **把 SAT gate 显式二阶段化**
   - 先单独判：是否通过最低满意线
   - 再只对 SAT 样本做 `4/5` 细分

2. **把 DSAT 召回变成 prompt 的硬目标**
   - 明确要求先列出是否存在“关键失败”
   - 若存在关键失败，不允许 calibration 先验把它抬成 SAT

3. **限制 calibration 的作用范围**
   - calibration 只用于：
     - SAT 之后的 `4/5`
     - DSAT 之后的 `1/2/3`
   - 不用于决定是否通过 SAT gate

换句话说：

`v3.1` 已经说明“在单 prompt 里同时做 calibration 和强 SAT gate”还是容易互相干扰。  
下一步更合理的是把它拆成一个更明确的阶段式流程，而不是继续在单 prompt 上微调措辞。
