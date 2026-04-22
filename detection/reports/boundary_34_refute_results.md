# `boundary_34_refute` Prompt 结果分析

## 实验设置

- backbone: `Qwen/Qwen3-8B`
- memory update mode: `none`
- turn eval prompt version: `boundary_34_refute`
- result file: `outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_refute.jsonl`
- 对比基线：
  - `Qwen none (v2)`: `outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
  - `Qwen qwen_short`: `outputs/personalized/Qwen_Qwen3-8B_test_none_qwen_short.jsonl`
  - `Qwen boundary_34`: `outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34.jsonl`
  - `Qwen none + MS`: `outputs/personalized/Qwen_Qwen3-8B_test_none_calMS.jsonl`
  - `Qwen none + CDF`: `outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl`

补充对比 JSON：

- `outputs/personalized/boundary_34_refute_comparison.json`

评测命令：

```bash
cd detection
PYTHONPATH=. python eval/personalized.py \
  --result_files \
    qwen_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl \
    qwen_short=outputs/personalized/Qwen_Qwen3-8B_test_none_qwen_short.jsonl \
    qwen_boundary34=outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34.jsonl \
    qwen_boundary34_refute=outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_refute.jsonl \
    qwen_none_ms=outputs/personalized/Qwen_Qwen3-8B_test_none_calMS.jsonl \
    qwen_none_cdf=outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl \
  --output_json outputs/personalized/boundary_34_refute_comparison.json
```

---

## 主要结论

`boundary_34_refute` 没有成为更强的整体方案，但它比 `boundary_34` 更接近真正想要的方向：

- 它**显著压低了 `false_sat_rate`**
- 同时**提升了 `F1-DSAT`**
- 但代价是：
  - 过度把真实满意判成不满意
  - 导致全局 1-5 指标和边界 Accuracy 明显下降

更具体地说：

- 相比 `boundary_34`，`boundary_34_refute` 的“反证式 3/4 判别”是有效的
- 相比 `Qwen none`，它确实更会抓 DSAT
- 但它已经从“默认判满意”摆到了另一个极端：
  - **过于保守，开始大量错杀 SAT**

所以当前更准确的判断是：

- `boundary_34_refute` 证明了 **failure-first / refutation** 这个方向有价值
- 但这版 prompt 本身 **过强地鼓励判 3**，还不能直接作为主线系统

---

## 全局指标

| 方法 | MAE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|
| Qwen none | 0.7110 | 0.2967 | 0.2820 | 0.2815 |
| Qwen qwen\_short | 0.6752 | 0.2599 | 0.2462 | 0.2369 |
| Qwen boundary\_34 | 0.7363 | 0.1449 | 0.1062 | 0.0774 |
| Qwen boundary\_34\_refute | 0.8482 | 0.1978 | 0.1752 | 0.1270 |
| Qwen none + MS | 0.6277 | 0.3668 | 0.3674 | 0.3589 |
| Qwen none + CDF | 0.6355 | 0.3601 | 0.3716 | 0.3595 |

结论：

- `boundary_34_refute` 的全局 1-5 指标明显劣化。
- 这是预期的一部分，因为它只输出 `3/4`；但即使在只看粗粒度排序时，也没有显示出比 `Qwen none` 更稳定的整体质量。
- 因此，这版不能作为“更好的通用满意度预测器”。

---

## 3/4 边界指标

| 方法 | Acc | F1-macro | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen none | 0.7555 | 0.5908 | 0.8504 | 0.3312 | 0.1824 | 0.6444 | 0.6459 | 0.1617 |
| Qwen qwen\_short | 0.7990 | 0.5641 | 0.8841 | 0.2440 | 0.1390 | 0.6181 | 0.8103 | 0.0753 |
| Qwen boundary\_34 | 0.7912 | 0.5525 | 0.8793 | 0.2257 | 0.1149 | 0.5478 | 0.8220 | 0.0824 |
| Qwen boundary\_34\_refute | 0.6793 | 0.5649 | 0.7880 | 0.3418 | 0.1540 | 0.6030 | 0.5131 | 0.2810 |
| Qwen none + MS | 0.7848 | 0.6126 | 0.8709 | 0.3542 | 0.2252 | 0.6823 | 0.6549 | 0.1245 |
| Qwen none + CDF | 0.7930 | 0.6209 | 0.8763 | 0.3655 | 0.2422 | 0.6759 | 0.6513 | 0.1153 |

重点对比：

### 相比 `boundary_34`

- `F1-DSAT: 0.2257 -> 0.3418`
- `false_sat_rate: 0.8220 -> 0.5131`
- `AUC: 0.5478 -> 0.6030`
- 但：
  - `Accuracy: 0.7912 -> 0.6793`
  - `F1-SAT: 0.8793 -> 0.7880`
  - `false_dsat_rate: 0.0824 -> 0.2810`

这说明：

- 反证式 prompt 确实成功抑制了“默认判满意”
- 但它压得太过，开始大面积把 SAT 拉回 DSAT

### 相比 `Qwen none`

- `F1-DSAT: 0.3312 -> 0.3418`，有小幅提升
- `false_sat_rate: 0.6459 -> 0.5131`，有明显改善
- 但：
  - `Accuracy: 0.7555 -> 0.6793`
  - `AUC: 0.6444 -> 0.6030`
  - `false_dsat_rate: 0.1617 -> 0.2810`

因此这版的真实形态不是“边界整体变好”，而是：

- **DSAT 检出更强**
- **SAT 误伤也更强**

---

## 用户感知二分类指标

| 方法 | PU-Acc | PU-F1-macro | PU-F1-SAT | PU-F1-DSAT | PU-Kappa | PU-AUC | WC-bin-Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen none | 0.7430 | 0.5489 | 0.8244 | 0.2734 | 0.1196 | 0.6164 | 0.1529 |
| Qwen qwen\_short | 0.7863 | 0.5280 | 0.8663 | 0.1897 | 0.0930 | 0.5976 | 0.1368 |
| Qwen boundary\_34 | 0.7799 | 0.5294 | 0.8638 | 0.1951 | 0.0972 | 0.5568 | 0.1129 |
| Qwen boundary\_34\_refute | 0.6791 | 0.5385 | 0.7747 | 0.3024 | 0.1267 | 0.6015 | 0.1487 |
| Qwen none + MS | 0.7691 | 0.5454 | 0.8371 | 0.2537 | 0.1099 | 0.5748 | 0.1334 |
| Qwen none + CDF | 0.7741 | 0.5380 | 0.8449 | 0.2311 | 0.0825 | 0.5745 | 0.1337 |

这里的信号更有价值。

### 相比 `boundary_34`

- `PU-F1-DSAT: 0.1951 -> 0.3024`
- `PU-Kappa: 0.0972 -> 0.1267`
- `PU-AUC: 0.5568 -> 0.6015`
- `WC-bin-Pearson: 0.1129 -> 0.1487`

这说明：

- 在“每个用户内部是否更会抓不满意”这件事上，`boundary_34_refute` 明显比 `boundary_34` 好

### 相比 `Qwen none`

- `PU-F1-DSAT: 0.2734 -> 0.3024`
- `PU-Kappa: 0.1196 -> 0.1267`
- `PU-AUC: 0.6164 -> 0.6015`（略降）
- `WC-bin-Pearson: 0.1529 -> 0.1487`（略降）

这说明：

- `boundary_34_refute` 在用户级 DSAT 捕捉上确实有增益
- 但用户级整体边界建模并没有全面超过 `Qwen none`

因此更准确的表述是：

- 它改善了“抓不满意”的能力
- 但没有把“抓不满意”和“保住满意”同时做好

---

## 预测分布与误差形态

预测分布：

- `pred=3`: `2047`
- `pred=4`: `4427`

相比 `boundary_34`：

- `boundary_34`: `pred=3 639`, `pred=4 5835`
- `boundary_34_refute`: `pred=3 2047`, `pred=4 4427`

这说明 `refute` 的确把模型从“几乎总判 4”拉回来了。

关键混淆：

- `gold=3 -> pred=4`: `415`
- `gold=4 -> pred=3`: `790`
- `gold=5 -> pred=3`: `718`
- `gold=5 -> pred=4`: `2200`

和 `boundary_34` 相比：

- `3 -> 4` 明显减少
- 但 `4 -> 3`、`5 -> 3` 明显增加

所以这版本质上是：

- **用更低的 false SAT**
- 换来了
- **更高的 false DSAT**

---

## 分任务观察

各任务边界表现：

| 任务 | Acc | F1-DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|
| 技能学习规划 | 0.7147 | 0.3043 | 0.6348 | 0.2133 |
| 旅行规划 | 0.6630 | 0.3609 | 0.4829 | 0.3041 |
| 礼物准备 | 0.6328 | 0.3450 | 0.4540 | 0.3485 |
| 菜谱规划 | 0.7292 | 0.3386 | 0.5224 | 0.2280 |

观察：

- `旅行规划` 和 `礼物准备` 上，`false_sat_rate` 压得最好
- 但同时这两个任务的 `false_dsat_rate` 也最高

这说明：

- 反证式 prompt 对“开放任务”更容易触发保守判断
- 一旦回复没有显式覆盖所有关键点，模型就更愿意把它拉回 `3`

---

## 结论

`boundary_34_refute` 的价值在于：

1. 它验证了 `failure-first / refutation` 方向确实能抑制“默认判满意”。
2. 它比 `boundary_34` 更接近真正的 `3/4` 边界目标。
3. 它在用户级 DSAT 指标上有真实增益。

但它的问题也很明确：

1. 它过度保守，明显提高了 `false_dsat_rate`。
2. 它没有成为比 `Qwen none` 更好的整体边界系统。
3. 它仍然明显弱于 `Qwen none + calibration` 这类后处理方案。

因此当前最合理的结论是：

- **这不是最终可用版本**
- 但它提供了一个重要信号：
  - 继续优化时，确实应该保留“先找降分证据”的机制
  - 但要削弱“只要有一点缺口就判 3”的强度

---

## 对下一步设计的启示

下一版如果继续沿这个方向推进，最值得改的不是大框架，而是“反证门槛”本身：

1. 把 `3` 的触发条件收紧  
   只在“存在明确且关键的失败”时才判 `3`，不要把普通的“不够细致”自动当成降线证据。

2. 明确区分：
   - `未达满意线`
   - `达到满意线但不够优秀`

3. 对 `4` 增加一个更明确的保护条件  
   例如：
   - 只要核心问题已回答
   - 关键约束已满足
   - 剩余缺口仅属于“可改进但不致命”
   就应优先判 `4`

4. 如果继续保留反证式 prompt，建议只要求极短输出  
   当前长分析会放大模型的保守倾向，也拖慢推理。

5. 更适合的下一步可能是：
   - 保留 `refute` 的 failure-first 逻辑
   - 但改成一个更短、更硬的二段式 prompt
   - 或只对初判接近 `3/4` 边界的样本触发 `refute` 二判，而不是全量使用
