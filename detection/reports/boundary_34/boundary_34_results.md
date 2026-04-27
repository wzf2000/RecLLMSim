# `boundary_34` Prompt 结果分析

## 实验设置

- backbone: `Qwen/Qwen3-8B`
- memory update mode: `none`
- turn eval prompt version: `boundary_34`
- result file: `outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34.jsonl`
- 对比基线：
  - `Qwen none (v2)`: `outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
  - `Qwen qwen_short`: `outputs/personalized/Qwen_Qwen3-8B_test_none_qwen_short.jsonl`
  - `Qwen none + MS`: `outputs/personalized/Qwen_Qwen3-8B_test_none_calMS.jsonl`
  - `Qwen none + CDF`: `outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl`

评测命令：

```bash
cd detection
PYTHONPATH=. python eval/personalized.py \
  --result_files \
    qwen_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl \
    qwen_short=outputs/personalized/Qwen_Qwen3-8B_test_none_qwen_short.jsonl \
    qwen_boundary34=outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34.jsonl \
    qwen_none_ms=outputs/personalized/Qwen_Qwen3-8B_test_none_calMS.jsonl \
    qwen_none_cdf=outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl \
  --output_json outputs/personalized/boundary_34_comparison.json
```

---

## 主要结果

### 1. 全局指标

| 方法 | MAE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|
| Qwen none | 0.7110 | 0.2967 | 0.2820 | 0.2815 |
| Qwen qwen\_short | 0.6752 | 0.2599 | 0.2462 | 0.2369 |
| Qwen boundary\_34 | 0.7363 | 0.1449 | 0.1062 | 0.0774 |
| Qwen none + MS | 0.6277 | 0.3668 | 0.3674 | 0.3589 |
| Qwen none + CDF | 0.6355 | 0.3601 | 0.3716 | 0.3595 |

结论：

- `boundary_34` 的全局 1-5 分预测显著退化。
- 这符合设计预期的一部分，因为它本来就不再做 1/2/5 的细分；但从结果看，退化幅度过大，说明它不仅失去了细粒度能力，也没有换来更好的 3/4 边界判断。

### 2. 3/4 边界指标

| 方法 | Acc | F1-macro | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen none | 0.7555 | 0.5908 | 0.8504 | 0.3312 | 0.1824 | 0.6444 | 0.6459 | 0.1617 |
| Qwen qwen\_short | 0.7990 | 0.5641 | 0.8841 | 0.2440 | 0.1390 | 0.6181 | 0.8103 | 0.0753 |
| Qwen boundary\_34 | 0.7912 | 0.5525 | 0.8793 | 0.2257 | 0.1149 | 0.5478 | 0.8220 | 0.0824 |
| Qwen none + MS | 0.7848 | 0.6126 | 0.8709 | 0.3542 | 0.2252 | 0.6823 | 0.6549 | 0.1245 |
| Qwen none + CDF | 0.7930 | 0.6209 | 0.8763 | 0.3655 | 0.2422 | 0.6759 | 0.6513 | 0.1153 |

结论：

- `boundary_34` 没有提升最关键的 `F1-DSAT`，反而从 `0.3312` 降到 `0.2257`。
- `false_sat_rate` 恶化到 `0.8220`，说明它把更多真实不满意误判成了满意。
- 它在边界上甚至略差于 `qwen_short`，后者的 `F1-DSAT=0.2440`，`false_sat_rate=0.8103`。
- 如果允许 calibration，`Qwen none + CDF` 仍然远强于 `boundary_34`。

### 3. 用户感知二分类指标

| 方法 | PU-Acc | PU-F1-macro | PU-F1-SAT | PU-F1-DSAT | PU-Kappa | PU-AUC | WC-bin-Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen none | 0.7430 | 0.5489 | 0.8244 | 0.2734 | 0.1196 | 0.6164 | 0.1529 |
| Qwen qwen\_short | 0.7863 | 0.5280 | 0.8663 | 0.1897 | 0.0930 | 0.5976 | 0.1368 |
| Qwen boundary\_34 | 0.7799 | 0.5294 | 0.8638 | 0.1951 | 0.0972 | 0.5568 | 0.1129 |
| Qwen none + MS | 0.7691 | 0.5454 | 0.8371 | 0.2537 | 0.1099 | 0.5748 | 0.1334 |
| Qwen none + CDF | 0.7741 | 0.5380 | 0.8449 | 0.2311 | 0.0825 | 0.5745 | 0.1337 |

结论：

- 如果关注“每个用户内部是否准确判断满意/不满意”，`boundary_34` 同样没有收益。
- 相比 `Qwen none`：
  - `PU-F1-DSAT: 0.2734 -> 0.1951`
  - `PU-Kappa: 0.1196 -> 0.0972`
  - `WC-bin-Pearson: 0.1529 -> 0.1129`
- 它提升的主要仍是 SAT 多数类命中，而不是 DSAT 边界建模。

---

## 误差形态

### 1. 预测分布明显塌缩到 `4`

`boundary_34` 的预测分布为：

- `pred=3`: `639`
- `pred=4`: `5835`

而 gold 的二分类分布映射后为：

- `gold<=3 (DSAT)`: `1107`
- `gold>=4 (SAT)`: `5367`

这说明模型明显偏向输出 `4`，形成 SAT 偏置。

### 2. 关键混淆

主要错误包括：

- `gold=3 -> pred=4`: `639`
- `gold=4 -> pred=3`: `219`
- `gold=5 -> pred=3`: `223`
- `gold=5 -> pred=4`: `2695`

最关键的是第一项：所有真实 3 分样本里，大量被直接推到 4 分，导致 `false_sat_rate` 很高。

### 3. 分任务观察

各任务的 `false_sat_rate`：

- `技能学习规划`: `0.9087`
- `旅行规划`: `0.8200`
- `礼物准备`: `0.7301`
- `菜谱规划`: `0.8756`

结论：

- `boundary_34` 在所有任务上都偏向“判满意”。
- `技能学习规划` 和 `菜谱规划` 最严重，说明当任务更开放、回复较长或更容易“看起来像在帮忙”时，prompt 更容易把未达线的回复也判成满意。

---

## 结论

`boundary_34` 这个版本目前不值得作为后续主线，原因很明确：

1. 它没有提升真正关心的 DSAT 边界能力。
2. 它显著增加了 SAT 偏置，导致 `false_sat_rate` 进一步恶化。
3. 它不仅丢失了 1-5 的细粒度排序能力，也没有换来更好的用户级满意/不满意判断。

更直接地说：

- 把任务形式上收缩到 “只输出 3 或 4” 并不等于模型真的学会了 `3/4` 边界。
- 对 Qwen3-8B 来说，这种 prompt 反而更容易退化成：
  - “只要回复看起来有帮助，就给 4”
  - 而不是严格对照用户的最低满意线

---

## 对下一步设计的启示

如果还要继续沿着 `3/4` 主线推进，下一版不应该继续只靠“强行限制输出空间”，而应补上更强的负例约束：

- 显式要求先检查“哪些关键要求没有满足”
- 强化 `three_vs_four_distinction` 中的失败条件，而不是只描述“达标条件”
- 引入 `3分案例 vs 4分案例` 的对比式 anchor，而不是抽象地讲“满意最低线”
- 对预测为 `4` 的样本增加一层反证检查：
  - “有没有任何一个关键缺陷足以把它拉回 3 分？”

当前更合理的判断是：

- `Qwen none` 仍然是 raw setting 下更强的 3/4 边界基线
- 真正有效的提升仍主要来自 calibration，而不是当前这个 `boundary_34` prompt
