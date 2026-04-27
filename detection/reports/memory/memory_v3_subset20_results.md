# Memory V3 Subset20 Results

## 1. Setup

本次分析文件：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_u20.jsonl`

规模：

- `20` 个用户
- `1594` 个 assistant turns

对照方法：

- `Qwen none`
- `Qwen no_memory`
- `boundary_34_selective_refute_v2`

注意：下面主对比全部在**同一批 1594 个 sample_id**上重算。

## 2. Main Metric Comparison

| Method | MAE | Pearson | Spearman | QWK | Acc | F1-DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `memv3` | `0.6349` | `0.3163` | `0.3176` | `0.2687` | `0.8168` | `0.1705` | `0.8969` | `0.0238` |
| `Qwen none` | `0.7077` | `0.2999` | `0.3012` | `0.2807` | `0.7629` | `0.3505` | `0.6495` | `0.1450` |
| `boundary_v2` | `0.7955` | `0.2228` | `0.2123` | `0.1442` | `0.7315` | `0.3495` | `0.6048` | `0.1934` |
| `Qwen no_memory` | `0.7917` | `0.1392` | `0.1069` | `0.0706` | `0.8174` | `0.0459` | `0.9759` | `0.0054` |

直接结论：

- `memv3` 的 **全局 1-5 指标很强**
  - `MAE` 最好
  - `Pearson / Spearman` 最好
  - `QWK` 也明显高于 `boundary_v2`
- 但 `memv3` 的 **DSAT 边界能力明显变差**
  - `F1-DSAT = 0.1705`
  - `false_sat_rate = 0.8969`

所以这版不是“全面更好”，而是：

- **用更强的整体校准，换掉了对不满意样本的敏感性**

## 3. Distribution Diagnosis

真实分布：

- `1: 38`
- `2: 72`
- `3: 181`
- `4: 611`
- `5: 692`

`memv3` 预测分布：

- `2: 6`
- `3: 55`
- `4: 897`
- `5: 636`

没有任何 `1` 分预测。

SAT / DSAT：

- gold: `SAT=1303`, `DSAT=291`
- pred: `SAT=1533`, `DSAT=61`

这说明：

- `memv3` 显著偏 SAT
- 低分样本被大量压回 `4`
- 尤其 `1/2/3` 区间几乎塌掉，只剩极少量 `2/3`

与 `Qwen none` 对比：

- `Qwen none` pred: `{1: 1, 2: 23, 3: 267, 4: 924, 5: 379}`
- `memv3` pred: `{2: 6, 3: 55, 4: 897, 5: 636}`

也就是说，`memv3` 的主要变化不是“整体更准地分 1-5”，而是：

- 把一大批 `3/4` 和一部分 `4` 往 `5` 推
- 同时几乎不愿意给低分

## 4. Reason Consistency

这版 reason 语义是完全合法的：

- `pred_score >= 4` 且 `reason != 满意`: `0`
- `pred_score <= 3` 且 `reason == 满意`: `0`

所以这次问题不在 reason，而在分数分布本身。

## 5. Memory Usage Diagnosis

本次最关键的发现是：`memv3` **确实更强地用了 memory**，而且是按我们预期的“校准刻度”在用。

### 5.1 Block-level correlation

按 block 比较 `memory.avg_satisfaction_score` 与 block 预测均分的相关性：

- `Qwen none`: `0.5566`
- `boundary_v2`: `0.2127`
- `memv3`: `0.6305`

再看 block 预测均分与真实 block 均分的相关性：

- `Qwen none`: `0.5171`
- `boundary_v2`: `0.2951`
- `memv3`: `0.6083`

这说明：

- `memv3` 显著增强了 memory 的“用户级均值校准”作用
- 而且这种增强方向是对的，因为它更接近真实 block 均值

### 5.2 Strictness buckets

按 memory 平均分把 block 分成三组：

#### strict

- `n = 210`
- gold mean = `3.676`
- pred mean = `4.124`
- gold SAT = `0.671`
- pred SAT = `0.910`

#### mid

- `n = 702`
- gold mean = `3.923`
- pred mean = `4.226`
- gold SAT = `0.738`
- pred SAT = `0.954`

#### lenient

- `n = 682`
- gold mean = `4.550`
- pred mean = `4.563`
- gold SAT = `0.944`
- pred SAT = `0.985`

解释很明确：

- 对宽松用户，`memv3` 的刻度基本合理
- 对严格用户和中间用户，`memv3` 仍然明显高估 SAT

这说明 `v3` 虽然比 `v2` 更强调 calibration，但目前的 prompt 把“证据不足时保守”几乎解释成了：

- **尽量不给低分**

于是产生了当前这种：

- 全局 MAE 变好
- 但 DSAT 召回严重下降

## 6. Interpretation

结合前面的 memory cache 分析，这版结果很符合预期中的一种失败模式：

### 6.1 正向部分：我们的判断是对的

之前对 `v2` 的判断是：

- Qwen3 的 memory 真正强项是 calibration
- boundary/fullscale 更容易被弱边界规则带偏

`memv3` 的结果证明这个判断基本正确：

- 它确实恢复了更强的 calibration 效应
- 全局 1-5 指标明显改善

### 6.2 负向部分：v3 目前削弱边界削得过头了

当前 `v3` 把这些策略同时做了：

- 弱化低证据边界
- 低分证据 sparse/none 时默认更保守
- 先使用 calibration 先验

这些方向 individually 都合理，但叠加后效果变成：

- 模型几乎总能为 SAT 找到更合理的解释
- 而 DSAT 只在特别严重时才会保留下来

于是：

- `false_dsat_rate` 很低：`0.0238`
- 但 `false_sat_rate` 极高：`0.8969`

这已经非常接近 `no_memory` 那种“默认满意”的失败模式，只是分数刻度比 `no_memory` 更像样。

## 7. Bottom Line

这版 `memv3` 的结论可以压成一句话：

**memory_v3 成功增强了 Qwen3 对用户级打分刻度的利用，但当前实现把“保守使用弱证据边界”推得太远，导致模型几乎失去抓 DSAT 的能力。**

所以它不是一个失败的方向，而是一个：

- 证明 calibration 分离思路是有效的
- 但目前需要把 DSAT 约束重新加回来

## 8. Next-Step Recommendation

下一步我不建议回退到 `v2`，而是基于当前 `v3` 继续做两处修正：

1. **不要把“低分证据 sparse/none”解释成“默认给 SAT”**
   - 当前 prompt 里“默认先给 3，只有严重错误才给 2/1”这条还不够
   - 还需要显式补一句：
     - `证据不足` 只影响 `1/2/3` 的内部细分
     - **不影响 `3/4` 边界本身**

2. **重新强化 3/4 最低满意线**
   - `calibration` 应该决定整体刻度
   - 但不能压过“是否过最低满意线”这个判断
   - 也就是：
     - `Step A` 先做 calibration
     - 但 `Step B` 必须是强 gate，而不是可被上一步轻易覆盖

如果继续沿这条线迭代，我认为最合理的是做一个 `v3.1`：

- 保留当前 calibration 分离
- 保留 evidence sufficiency
- 但把 `3/4` gate 明显加硬，让模型先决定是否过最低满意线，再决定是否因为 evidence sparse 而收缩到 `3` 而不是 `1/2`
