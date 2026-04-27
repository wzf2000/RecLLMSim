# 3/4 边界（满意/不满意）补充评测报告

> 本报告将原有 1-5 分满意度预测结果重新映射为二分类 SAT/DSAT：
> - `SAT = score >= 4`
> - `DSAT = score <= 3`
>
> 重点关注用户真正是否“满意”的判断能力，尤其是 `3/4` 边界，而非 `4/5` 的细粒度区分。

---

## 1. 本次补充了哪些指标

已将 SAT/DSAT 边界指标直接并入 `eval/personalized.py`，新增输出包括：

- 全局二分类指标：
  - `accuracy`
  - `f1_macro`
  - `f1_sat`
  - `f1_dsat`
  - `kappa`
  - `auc`
  - `false_sat_rate`
  - `false_dsat_rate`
- 用户感知二分类指标：
  - `pu_bin_accuracy`
  - `pu_bin_f1_sat`
  - `pu_bin_f1_dsat`
  - `pu_bin_kappa`
  - `wc_bin_pearson`

说明：

- `false_sat_rate`：真实不满意（<=3）却被预测成满意（>=4）的比例  
  这是当前任务中最值得重点压低的错误类型。
- `false_dsat_rate`：真实满意（>=4）却被预测成不满意（<=3）的比例。

补充汇总 JSON：

- `outputs/personalized/boundary_comparison_all.json`

---

## 2. 主要方法在边界上的表现

### 2.1 GPT 系列（raw）

| 方法 | Acc↑ | F1-SAT↑ | F1-DSAT↑ | Kappa↑ | AUC↑ | False SAT↓ |
|---|---:|---:|---:|---:|---:|---:|
| GPT no\_memory | 0.8231 | 0.9019 | 0.1048 | 0.0616 | 0.5441 | 0.9395 |
| GPT v1 none | 0.7260 | 0.8314 | 0.2694 | 0.1023 | 0.6062 | 0.7046 |
| GPT v1 per\_session | 0.7320 | 0.8353 | 0.2816 | 0.1183 | 0.6081 | 0.6929 |
| GPT v1 oracle | 0.7288 | 0.8325 | 0.2868 | 0.1214 | 0.6149 | 0.6811 |
| GPT v2 none | 0.7935 | 0.8796 | 0.2730 | 0.1584 | 0.6397 | 0.7733 |
| GPT v2 per\_session | 0.7913 | 0.8779 | 0.2818 | 0.1641 | 0.6397 | 0.7606 |
| GPT v2 oracle | **0.7939** | 0.8790 | **0.3059** | **0.1882** | **0.6531** | **0.7344** |

结论：

- GPT 的 raw 结果里，`v2 oracle` 是边界表现最好的版本。
- 但 GPT 整体的一个明显问题是：**很容易把真实不满意判成满意**。
- 例如 `GPT no_memory` 虽然 Accuracy 很高，但 `false_sat_rate=0.9395`，几乎把所有不满意都漏掉了，因此这个高 Accuracy 没有实际意义。

### 2.2 Qwen 系列（raw）

| 方法 | Acc↑ | F1-SAT↑ | F1-DSAT↑ | Kappa↑ | AUC↑ | False SAT↓ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen no\_memory | 0.8289 | 0.9057 | 0.0720 | 0.0483 | 0.5535 | 0.9612 |
| Qwen none | 0.7555 | 0.8504 | **0.3312** | 0.1824 | **0.6444** | **0.6459** |
| Qwen per\_session | 0.7610 | 0.8547 | 0.3271 | 0.1821 | 0.6437 | 0.6603 |
| Qwen oracle | 0.7651 | 0.8578 | 0.3249 | **0.1827** | 0.6392 | 0.6694 |
| Qwen per\_turn | 0.7631 | 0.8569 | 0.3115 | 0.1684 | 0.6364 | 0.6865 |
| Qwen short | **0.7990** | **0.8841** | 0.2440 | 0.1390 | 0.6181 | 0.8103 |

结论：

- 如果只看边界任务，**Qwen raw 明显强于 GPT raw**，尤其是 `F1-DSAT` 和 `false_sat_rate`。
- `Qwen none` / `Qwen per_session` / `Qwen oracle` 三者非常接近，其中：
  - `Qwen none` 的 `F1-DSAT` 最好（`0.3312`）
  - `Qwen none` 的 `false_sat_rate` 也最低（`0.6459`）
- `qwen_short` 再次体现出“高 Accuracy 但边界变差”的特征：
  - Accuracy 最高（`0.7990`）
  - 但 `F1-DSAT` 下降到 `0.2440`
  - `false_sat_rate` 恶化到 `0.8103`

这说明 `qwen_short` 进一步强化了“把更多样本判为满意”的倾向，因此**不适合作为 3/4 边界主线**。

---

## 3. 用户感知边界指标

### 3.1 GPT 系列（raw）

| 方法 | PU-Acc↑ | PU-F1-SAT↑ | PU-F1-DSAT↑ | PU-Kappa↑ | WC-Pearson↑ |
|---|---:|---:|---:|---:|---:|
| GPT no\_memory | **0.8082** | **0.8849** | 0.0983 | 0.0610 | 0.0993 |
| GPT v1 none | 0.7163 | 0.8134 | 0.2446 | 0.0922 | 0.1213 |
| GPT v1 per\_session | 0.7246 | 0.8191 | 0.2495 | 0.1034 | 0.1351 |
| GPT v1 oracle | 0.7204 | 0.8151 | 0.2577 | 0.1056 | 0.1340 |
| GPT v2 none | 0.7757 | **0.8513** | 0.2023 | 0.0785 | 0.1055 |
| GPT v2 per\_session | 0.7745 | 0.8487 | 0.2019 | 0.0736 | 0.0960 |
| GPT v2 oracle | **0.7772** | 0.8501 | **0.2290** | **0.0986** | **0.1152** |

结论：

- GPT 的用户级边界能力并不突出，尤其 `PU-F1-DSAT` 整体偏低。
- `v2 oracle` 在 GPT 内部仍然最好，但提升不大。

### 3.2 Qwen 系列（raw）

| 方法 | PU-Acc↑ | PU-F1-SAT↑ | PU-F1-DSAT↑ | PU-Kappa↑ | WC-Pearson↑ |
|---|---:|---:|---:|---:|---:|
| Qwen no\_memory | **0.8138** | **0.8894** | 0.0840 | 0.0655 | 0.1192 |
| Qwen none | 0.7430 | 0.8244 | 0.2734 | 0.1196 | 0.1529 |
| Qwen per\_session | 0.7495 | 0.8316 | **0.2822** | **0.1378** | **0.1574** |
| Qwen oracle | 0.7527 | **0.8356** | 0.2728 | 0.1303 | 0.1554 |
| Qwen per\_turn | 0.7511 | 0.8333 | 0.2610 | 0.1162 | 0.1470 |
| Qwen short | **0.7863** | **0.8663** | 0.1897 | 0.0930 | 0.1368 |

结论：

- 如果看真正的“每个用户内部是否判对满意/不满意”，**Qwen per_session 是 raw 方案里最好的**：
  - `PU-F1-DSAT = 0.2822`
  - `PU-Kappa = 0.1378`
  - `WC-Pearson = 0.1574`
- `Qwen none` 和 `Qwen oracle` 也很接近。
- `qwen_short` 虽然 `PU-Acc` 高，但 `PU-F1-DSAT` 明显掉到 `0.1897`，说明它更像是在“多数类 SAT”上讨巧，而不是更好地区分 DSAT。

---

## 4. 校准方法对边界任务的影响

### 4.1 GPT v2 + calibration

| 方法 | Acc↑ | F1-DSAT↑ | Kappa↑ | AUC↑ | False SAT↓ |
|---|---:|---:|---:|---:|---:|
| GPT v2 none + MS | 0.7953 | 0.3008 | 0.1849 | 0.6538 | 0.7425 |
| GPT v2 none + CDF | 0.7774 | 0.3441 | 0.2101 | 0.6615 | 0.6585 |
| GPT v2 oracle + MS | 0.7963 | 0.3225 | 0.2053 | 0.6689 | 0.7164 |
| GPT v2 oracle + CDF | 0.7813 | **0.3546** | **0.2229** | **0.6726** | **0.6486** |

结论：

- 对 GPT 来说，**CDF 校准明显更适合边界任务**，尤其能提升 `F1-DSAT` 和降低 `false_sat_rate`。
- `GPT v2 oracle + CDF` 是 GPT 全部方法里最好的边界版本。

### 4.2 Qwen + calibration

| 方法 | Acc↑ | F1-DSAT↑ | Kappa↑ | AUC↑ | False SAT↓ |
|---|---:|---:|---:|---:|---:|
| Qwen none + MS | 0.7848 | 0.3542 | 0.2252 | **0.6823** | 0.6549 |
| Qwen none + CDF | 0.7930 | 0.3655 | 0.2422 | 0.6759 | 0.6513 |
| Qwen oracle + MS | 0.7885 | 0.3459 | 0.2203 | 0.6758 | 0.6730 |
| Qwen oracle + CDF | **0.7936** | **0.3674** | **0.2445** | 0.6766 | **0.6495** |

结论：

- 对 Qwen 来说，**calibration 对 3/4 边界同样是显著增益**。
- 最好的边界方法是：
  - `Qwen oracle + CDF`：`F1-DSAT = 0.3674`，`Kappa = 0.2445`
  - `Qwen none + CDF` 也几乎一样强
- 与 raw Qwen 相比，校准后的 DSAT 识别能力提升明显。

---

## 5. 总结：如果目标是“满意 / 不满意”边界，哪些方法最好？

### 5.1 如果只看 raw 方法

我会这样排序：

1. **Qwen per_session / Qwen none / Qwen oracle**
   - 原因：`F1-DSAT`、`false_sat_rate`、用户感知边界指标都明显优于 GPT
   - 其中 `Qwen per_session` 的用户感知边界能力最好

2. **GPT v2 oracle**
   - 是 GPT raw 里的最好版本
   - 但仍弱于 Qwen raw 的边界表现

3. **qwen_short**
   - 不建议作为边界方向继续推进
   - 它更像是在牺牲 DSAT 识别换 Accuracy

### 5.2 如果允许 post-hoc calibration

最优方法是：

1. **Qwen oracle + CDF**
2. **Qwen none + CDF**
3. **Qwen none + MS**
4. **GPT v2 oracle + CDF**

也就是说，当前项目里若只问“这条回复用户到底满意还是不满意”，**Qwen + calibration 已经是最强组合**。

---

## 6. 对后续 system 设计的启示

这轮补充验证给出三个很明确的信号：

1. **你真正关心的 3/4 边界上，Qwen raw 已经比 GPT raw 更有潜力。**

2. **`qwen_short` 方向不对。**
   它提升的是多数类 SAT 的命中，不是用户级 DSAT 边界建模。

3. **真正值得继续优化的是：**
   - `DSAT` 的召回与 F1
   - `false_sat_rate`（把不满意误判成满意）
   - `PU-F1-DSAT / PU-Kappa / WC-Pearson`

因此，下一步 prompt / memory / anchor 设计应当围绕：

- “这个用户的满意最低线（3/4 边界）是什么？”
- “哪些缺陷会让回复跌破满意线？”
- “怎样减少把真实不满意判成满意？”

而不是继续围绕 `4/5` 或全局 MAE 做优化。

---

## 7. 复现方式

单文件：

```bash
cd detection
result_file=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl \
  bash scripts/eval_personalized.sh
```

多文件对比：

```bash
cd detection
PYTHONPATH=$(pwd) python eval/personalized.py \
  --result_files \
    qwen_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl \
    qwen_per_session=outputs/personalized/Qwen_Qwen3-8B_test_per_session.jsonl \
    qwen_short=outputs/personalized/Qwen_Qwen3-8B_test_none_qwen_short.jsonl \
  --output_json outputs/personalized/boundary_subset.json
```

本次完整汇总 JSON：

```text
outputs/personalized/boundary_comparison_all.json
```
