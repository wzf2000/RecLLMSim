# Boundary 3/4 Selective Refute V2 Reason-Fix Anchor1 Subset20 Results

## 1. Setup

本次实验比较：

- 基线：`boundary_34_selective_refute_v2 + reason-fix`
- 实验：`boundary_34_selective_refute_v2 + reason-fix + n_anchors=1`

结果文件：

- base:
  `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_reasonfix_u20.jsonl`
- anchor1:
  `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_reasonfix_anchor1_u20.jsonl`

两者均为：

- `20` 个用户
- `1594` 个 assistant turns

## 2. Sanity Check

两个文件都满足 reason 合法性规则：

- `pred_score >= 4 => reason_prediction = 满意`
- `pred_score <= 3 => reason_prediction != 满意`

非法组合数量：

- base: `0`
- anchor1: `0`

## 3. Main Comparison

| Metric | base | anchor1 | delta |
|---|---:|---:|---:|
| MAE | `0.7967` | `0.7779` | `-0.0188` |
| Pearson | `0.2215` | `0.2285` | `+0.0070` |
| QWK | `0.1436` | `0.1464` | `+0.0028` |
| Boundary Acc | `0.7302` | `0.7491` | `+0.0189` |
| F1-DSAT | `0.3524` | `0.3631` | `+0.0107` |
| False SAT | `0.5979` | `0.6082` | `+0.0103` |
| False DSAT | `0.1965` | `0.1711` | `-0.0254` |
| PU-bin F1-DSAT | `0.3027` | `0.3088` | `+0.0061` |
| PU-bin Kappa | `0.1449` | `0.1630` | `+0.0181` |
| WC-bin Pearson | `0.1705` | `0.1888` | `+0.0183` |

## 4. Distribution Change

真实分布不变：

- gold: `SAT=1303`, `DSAT=291`

预测分布：

- base: `SAT=1221`, `DSAT=373`
- anchor1: `SAT=1257`, `DSAT=337`

因此 `anchor1` 的直接效果是：

- 更少把样本判成 `3`
- 整体略向 `SAT` 方向移动

但这一移动并没有让边界指标整体变坏，相反：

- `Accuracy` 提升
- `F1-DSAT` 也提升
- `false_dsat_rate` 明显下降

代价是：

- `false_sat_rate` 小幅变差

## 5. Selective Diagnostics

### 5.1 Trigger rate decreased

base:

- `model_flag = 297`
- `triggered = 241`

anchor1:

- `model_flag = 237`
- `triggered = 211`

说明加入 `anchor1` 之后：

- first-pass 更少把样本判成“高度接近边界”
- selective second-pass 进一步退居辅助角色

### 5.2 Trigger composition

anchor1 触发样本：

- `initial_score=3`, `initial_reason=不够细致`: `237`
- `initial_score=4`, `initial_reason=满意`: `4`

和 base 相比，模式没变，只是触发总量下降。

### 5.3 Second pass is still light-weight

anchor1 的 second-pass 改判统计：

- `(3 -> 3)`: `231`
- `(3 -> 4)`: `6`
- `(4 -> 4)`: `4`

其中：

- `3 -> 4` 修正真实 SAT: `5`
- `3 -> 4` 误伤真实 DSAT: `1`
- `4 -> 3`: `0`

这说明 second-pass 仍然不是主增益来源，anchor 带来的主要变化仍然发生在 first-pass。

## 6. Interpretation

`anchor1` 的效果不是“全面大幅提升”，但它确实带来了一个**小而稳定的正收益**：

- 全局指标略好
- 二分类边界指标略好
- 用户感知二分类指标也略好

不过它的收益类型比较明确：

- 更像是在减少对真实 SAT 的误伤
- 而不是明显增强“不满意”的抓取

证据是：

- `false_dsat_rate` 明显下降
- 但 `false_sat_rate` 小幅上升

所以这版 anchor 更像一个“温和校准器”，而不是专门增强 DSAT recall 的机制。

## 7. Conclusion

当前结论可以压成一句：

**`n_anchors=1` 值得保留为一个可行变体，但还不构成强到足以直接替代 base 的证据。**

更具体地说：

- 它优于 base 的幅度不大，但方向基本是正的
- 它没有引入新的 reason 语义问题
- 它没有让 selective 机制失控
- 它的增益更偏“整体稳定性/校准”，不是明显增强个性化 DSAT 建模

## 8. Recommendation

如果继续试 anchor，我建议只再补一个非常小的增量实验：

- `n_anchors=2`

目的不是继续扫很多配置，而是回答一个具体问题：

- `anchor1` 的轻微正收益会不会在 `anchor2` 继续提升
- 还是会因为 prompt 变长和参考案例噪声而回落

如果 `anchor2` 不能继续带来稳定提升，就可以停止这条线，回到 first-pass 本身。
