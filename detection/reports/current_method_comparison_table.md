# Current Method Comparison Table

## 1. Why QWK Looks Very Different Across Experiments

当前项目里其实混合了两类不同任务：

### A. Full 1-5 score prediction

- 模型直接预测 `1/2/3/4/5`
- 全局 `QWK` 是这个任务下的主要指标之一

### B. 3/4 boundary prediction

- prompt 设计目标是判断“满意 / 不满意”
- 很多 boundary 版本只输出 `3/4`
- 这类方法再用 1-5 全局 `QWK` 评测时，数值会天然偏低，因为它们根本不尝试区分 `1/2/5`

因此：

- `personalized_satisfaction_results.md` 里的 `QWK≈0.28~0.32`
  主要对应 **完整 1-5 评分任务**
- boundary 系列里常见的 `QWK≈0.08~0.15`
  对应 **3/4 边界任务被拿去做 1-5 QWK 评测**

这两类数值**不能直接横比**。

---

## 2. Full-Run Methods: 1-5 Score Prediction

下面这张表只看“完整 1-5 评分任务”的全量结果（90 用户 / 6474 turns）。

| Method | Task Type | MAE | Pearson | Spearman | QWK |
|---|---|---:|---:|---:|---:|
| `Qwen no_memory` | 1-5 | `0.7471` | `0.1868` | `0.1184` | `0.1096` |
| `Qwen none` | 1-5 | `0.7110` | `0.2967` | `0.2820` | `0.2815` |
| `Qwen per_session` | 1-5 | `0.7133` | `0.2812` | `0.2619` | `0.2694` |
| `Qwen per_session_oracle` | 1-5 | `0.7141` | `0.2737` | `0.2563` | `0.2632` |
| `Qwen per_turn` | 1-5 | `0.7087` | `0.2794` | `0.2641` | `0.2694` |
| `Qwen qwen_short` | 1-5 | `0.6752` | `0.2599` | `0.2462` | `0.2369` |
| `Qwen none + MS` | 1-5 | `0.6277` | `0.3668` | `0.3674` | `0.3589` |
| `Qwen none + CDF` | 1-5 | `0.6355` | `0.3601` | `0.3716` | `0.3595` |

### Read this table like this

- 如果目标是**完整 1-5 评分质量**：
  - raw baseline 最强的是 `Qwen none`
  - 若允许 post-hoc calibration，最强的是 `Qwen none + MS/CDF`
- `qwen_short` 虽然 `MAE` 更低，但 `QWK`、相关系数和用户感知能力都弱于 `Qwen none`

---

## 3. Full-Run Methods: 3/4 Boundary Prediction

下面这张表只看“满意 / 不满意”边界任务的全量结果。

这里更重要的指标是：

- `F1-DSAT`
- `false_sat_rate`
- `boundary kappa`
- `PU-bin F1-DSAT`

| Method | Task Type | 1-5 QWK | Boundary Acc | F1-DSAT | Boundary Kappa | AUC | False SAT | False DSAT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | 1-5 result remapped to boundary | `0.2815` | `0.7555` | `0.3312` | `0.1824` | `0.6444` | `0.6459` | `0.1617` |
| `Qwen qwen_short` | boundary-relevant readout from 1-5 result | `0.2369` | `0.7990` | `0.2440` | `0.1390` | `0.6181` | `0.8103` | `0.0753` |
| `Qwen boundary_34` | boundary-only | `0.0774` | `0.7912` | `0.2257` | `0.1149` | `0.5478` | `0.8220` | `0.0824` |
| `Qwen boundary_34_refute` | boundary-only | `0.1270` | `0.6793` | `0.3418` | `0.1540` | `0.6030` | `0.5131` | `0.2810` |
| `Qwen boundary_34_selective_refute_v2` | boundary-only | `0.1229` | `0.7260` | `0.3177` | `0.1510` | `0.5859` | `0.6269` | `0.2012` |
| `Qwen reasonfix + anchor2` | boundary-only | `0.1147` | `0.7252` | `0.3197` | `0.1526` | `0.5873` | `0.6224` | `0.2031` |

### Read this table like this

- 如果目标是**边界任务本身**，不要把 `1-5 QWK` 当主指标。
- 这张表里：
  - `Qwen none` 仍是很强的 raw boundary baseline
  - `boundary_34` 明显失败，偏 SAT 太严重
  - `boundary_34_refute` 更会抓 DSAT，但误伤 SAT 过多
  - `selective_refute_v2` 是目前最平衡的 boundary 主线
  - `reasonfix + anchor2` 语义更干净，但性能上和 `selective_refute_v2` 基本持平

---

## 4. User-Aware Boundary Comparison

为了看“是否真的学到用户自己的满意边界”，更应该看用户感知二分类指标。

| Method | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|
| `Qwen none` | `0.2734` | `0.1196` | `0.1529` |
| `Qwen qwen_short` | `0.1897` | `0.0930` | `0.1368` |
| `Qwen boundary_34` | `0.1951` | `0.0972` | `0.1129` |
| `Qwen boundary_34_refute` | `0.3024` | `0.1267` | `0.1487` |
| `Qwen boundary_34_selective_refute_v2` | `0.2765` | `0.1175` | `0.1323` |
| `Qwen reasonfix + anchor2` | `0.2736` | `0.1190` | `0.1377` |

### Read this table like this

- 如果目标是“每个用户内部是否判对满意/不满意”：
  - `boundary_34_refute` 的 DSAT 抓取最强
  - 但它代价是 SAT 误伤太大，不够平衡
  - `selective_refute_v2` 和 `reasonfix + anchor2` 更像可用主线

---

## 5. Practical Reading Guide

如果你以后只想快速判断“一个新方法该和谁比”，可以直接按这个规则看：

### 你做的是完整 1-5 评分方法

和这些方法比：

- `Qwen none`
- `Qwen none + MS`
- `Qwen none + CDF`

主指标：

- `MAE`
- `Pearson`
- `Spearman`
- `QWK`

### 你做的是 3/4 满意边界方法

和这些方法比：

- `Qwen none`
- `boundary_34_refute`
- `boundary_34_selective_refute_v2`

主指标：

- `F1-DSAT`
- `false_sat_rate`
- `boundary kappa`
- `PU-bin F1-DSAT`
- `PU-bin kappa`

### 不建议做的比较

- 不要用 `boundary_34_*` 的 `QWK` 去和 `Qwen none` 的 `QWK=0.2815` 直接判断谁更强
- 不要把只输出 `3/4` 的方法当成完整 1-5 打分器来解读

---

## 6. Current Bottom Line

一句话总结当前格局：

- **完整 1-5 主线**：`Qwen none` 是 raw baseline，`MS/CDF` 校准后最强
- **3/4 boundary 主线**：`boundary_34_selective_refute_v2` 是目前最平衡的系统版本
- `reasonfix + anchor2` 没有明显超越 `selective_refute_v2`
- 后续更值得继续投入的方向仍然是：
  - **first-pass 的边界判断语言**
  - 而不是继续扩大 anchor 数量
