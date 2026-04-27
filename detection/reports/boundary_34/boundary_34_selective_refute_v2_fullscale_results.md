# Boundary 3/4 Selective Refute V2 Fullscale Results

## 1. Setup

本次分析的结果文件：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_fullscale.jsonl`

规模：

- `90` 个用户
- `6474` 个 assistant turns

用于对比的主要基线：

- `Qwen none`: `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- `boundary_34_selective_refute_v2`: `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2.jsonl`
- `boundary_34_selective_refute_v2 reasonfix + anchor2`: `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_reasonfix_anchor2.jsonl`

本次 fullscale pipeline 的结构是：

1. 先用 `boundary_34_selective_refute_v2` 做 `3/4` 路由
2. SAT 分支细化 `4/5`
3. DSAT 分支细化 `1/2/3`

因此它不是一个单 prompt 直接打 `1-5` 的系统，而是一个分层 pipeline。

## 2. Main Results

### 2.1 Global 1-5 metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7110` | `1.0122` | `0.2967` | `0.2820` | `0.2815` |
| `boundary_v2` | `0.8015` | `1.0320` | `0.1923` | `0.1688` | `0.1229` |
| `reasonfix+anchor2` | `0.8023` | `1.0382` | `0.1793` | `0.1504` | `0.1147` |
| `fullscale` | `0.8551` | `1.1795` | `0.1916` | `0.1717` | `0.1797` |

直接结论：

- 相比 `boundary_v2` 和 `reasonfix+anchor2`，`fullscale` 的 `QWK` 明显更高
- 但 `MAE / RMSE` 明显更差
- 和 `Qwen none` 相比，`fullscale` 在所有主要 `1-5` 指标上都没有超过 baseline

### 2.2 3/4 boundary metrics

| Method | Acc | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7555` | `0.3312` | `0.1824` | `0.6444` | `0.6459` | `0.1617` |
| `boundary_v2` | `0.7260` | `0.3177` | `0.1510` | `0.5859` | `0.6269` | `0.2012` |
| `reasonfix+anchor2` | `0.7252` | `0.3197` | `0.1526` | `0.5873` | `0.6224` | `0.2031` |
| `fullscale` | `0.7076` | `0.3309` | `0.1550` | `0.6106` | `0.5772` | `0.2337` |

这里的画像很典型：

- `fullscale` 的 `F1-DSAT` 基本追平 `Qwen none`
- `false_sat_rate` 明显更低，说明它更愿意抓不满意
- 但 `false_dsat_rate` 明显更高，说明它会误伤更多真实满意样本
- 最终 boundary `Accuracy` 低于 `Qwen none`

因此，它不是一个更平衡的 boundary 系统，而是一个 **偏 DSAT 的 fullscale 系统**。

## 3. User-Aware Results

### 3.1 User-aware global metrics

| Metric | `Qwen none` | `boundary_v2` | `fullscale` |
|---|---:|---:|---:|
| PU-Pearson | `0.1997` | `0.1602` | `0.1683` |
| PU-Spearman | `0.1798` | `0.1552` | `0.1568` |
| PU-Kappa | `0.1646` | `0.0910` | `0.1390` |
| WC-Pearson | `0.2154` | `0.1721` | `0.1750` |

### 3.2 User-aware binary metrics

| Metric | `Qwen none` | `boundary_v2` | `fullscale` |
|---|---:|---:|---:|
| PU-bin F1-DSAT | `0.2734` | `0.2765` | `0.2912` |
| PU-bin Kappa | `0.1196` | `0.1175` | `0.1266` |
| WC-bin Pearson | `0.1529` | `0.1323` | `0.1434` |

解读：

- 在用户级二分类边界上，`fullscale` 比 `boundary_v2` 略好
- 但它仍然没有形成对 `Qwen none` 的全面超越
- 换句话说，`fullscale` 主要保留了 boundary 主线里“更愿意抓 DSAT”的倾向，但还没有把这种倾向转化为稳定更强的个性化 `1-5` 预测

## 4. Distribution and Branch Diagnostics

### 4.1 Final prediction distribution

真实分布：

- `1: 123`
- `2: 251`
- `3: 733`
- `4: 2449`
- `5: 2918`

`fullscale` 预测分布：

- `1: 24`
- `2: 495`
- `3: 1203`
- `4: 3239`
- `5: 1513`

对应 SAT/DSAT：

- gold: `SAT=5367`, `DSAT=1107`
- pred: `SAT=4752`, `DSAT=1722`

所以这版有两个明显现象：

1. 整体 **低估满意、高估不满意**
2. `5` 被明显压缩到 `4`，而 `1` 非常少

### 4.2 Router and branch usage

branch 统计：

- `sat_45 = 4752`
- `dsat_123 = 1722`

router 输出统计：

- `router 4 = 4752`
- `router 3 = 1722`

这说明：

- 最终 SAT/DSAT 划分 **完全继承 router**
- refine 只做 branch 内细化，不会跨过 `3/4` 边界纠错

此外：

- `fullscale_router_triggered = 1102`
- `fullscale_router_applied = 1102`
- `fullscale_refine_applied = 6474`

即：

- router 内部 selective-refute 仍然较活跃
- 但从 fullscale 角度看，所有样本都会进入一次 refine

### 4.3 Router-to-final transition

fullscale 的转移模式非常干净：

- `(3 -> 1) = 24`
- `(3 -> 2) = 495`
- `(3 -> 3) = 1203`
- `(4 -> 4) = 3239`
- `(4 -> 5) = 1513`

这再次说明：

- router 决定 SAT/DSAT 侧别
- refine 只在本侧别内做 ordinal 细化

## 5. Branch-local Diagnosis

### 5.1 Overall branch accuracy

- `sat_45` branch overall exact-match accuracy: `0.4421`
- `dsat_123` branch overall exact-match accuracy: `0.1469`

但这个数字混入了 router 侧别错误后的样本，不能直接说明 refinement 本身弱到这个程度。

### 5.2 Accuracy when router side is correct

只在 router 侧别正确时看 branch refinement：

- `sat_45` 且 gold `>=4`: `4113` 样本，`4/5` exact-match accuracy = `0.5108`
- `dsat_123` 且 gold `<=3`: `468` 样本，`1/2/3` exact-match accuracy = `0.5406`

这说明 refinement 本身并不差到完全不可用，甚至两侧都在 `0.51 ~ 0.54` 这个量级。

真正更大的问题是：

- `sat_45` 分支里混入了 `639` 个真实 DSAT 样本
- `dsat_123` 分支里混入了 `1254` 个真实 SAT 样本

因此当前 fullscale 的主瓶颈不是第二层细化，而是：

- **第一层 router 的 SAT/DSAT 侧别错误会被后续 pipeline 放大**

## 6. Reason Validity

reason 语义是一致的：

- `pred_score >= 4` 且 `reason != 满意`: `0`
- `pred_score <= 3` 且 `reason == 满意`: `0`

所以这版不存在旧系统里那种大规模非法 `score/reason` 组合问题。

## 7. Interpretation

当前最合理的结论是：

**`boundary_34_selective_refute_v2_fullscale` 成功把 boundary 主线扩展成了一个可运行的完整 `1-5` pipeline，但它还不是新的最优主线。**

具体来说：

- 好处：
  - 终于能输出完整 `1-5`
  - `QWK` 明显高于 boundary-only 系列
  - `reason` 语义完全合法
  - branch refinement 本身有一定可用性
- 问题：
  - `MAE / RMSE` 比 `Qwen none` 更差
  - 最终仍明显偏向 DSAT
  - `5` 大量被压成 `4`
  - hard routing 让 router 错误无法被下游修复

所以它更像是：

- 一个合理的 fullscale 原型
- 而不是已经可以替代 `Qwen none` 的最终系统

## 8. Practical Conclusion

如果问题是：

- 这版 fullscale 值不值得继续沿着同一结构优化？

我的判断是：

- **值得继续，但下一步重点不能再只放在 branch refinement 上**

更值得优先改的方向是：

1. 让 router 输出更接近真实 SAT/DSAT 分布，减少当前 DSAT 偏置
2. 给 refinement 更多“纠偏”空间，而不是完全 hard route
3. 优先优化 `4/5` 分支，因为当前 `5 -> 4` 压缩非常明显

换句话说，当前 fullscale pipeline 的真正限制不是“第二层不会细化”，而是：

- 第一层 boundary router 仍然过强地决定了最终结果
- 这个结构对 router 偏差过于敏感
