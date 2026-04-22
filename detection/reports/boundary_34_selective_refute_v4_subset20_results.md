# Boundary 3/4 Selective Refute V4 Results on 20-User Subset

## 1. Setup

本轮分析对象：

- 方法：`boundary_34_selective_refute_v4`
- backbone：`Qwen/Qwen3-8B`
- memory update：`none`
- 文件：`detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v4_u20.jsonl`

子集规模：

- `20` 个用户
- `1594` 个 assistant turns
- `346` 个 block

真实分布：

- SAT (`>=4`) = `1303` (`81.7%`)
- DSAT (`<=3`) = `291` (`18.3%`)

所有对比都在同一批 `1594` 个 `sample_id` 上重算。

## 2. Main Results

### 2.1 V4 Metrics

- `MAE = 0.7848`
- `Pearson = 0.2173`
- `QWK = 0.1391`
- `Boundary Accuracy = 0.7422`
- `F1-DSAT = 0.3445`
- `false_sat_rate = 0.6289`
- `false_dsat_rate = 0.1750`

用户感知二分类指标：

- `PU-bin F1-DSAT = 0.2890`
- `PU-bin Kappa = 0.1353`
- `WC-bin Pearson = 0.1597`

预测分布：

- gold: `SAT=1303`, `DSAT=291`
- pred: `SAT=1258`, `DSAT=336`

### 2.2 Same-Subset Comparison

| Method | MAE | Pearson | QWK | Acc | F1-DSAT | False SAT | False DSAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7077` | `0.2999` | `0.2807` | `0.7629` | `0.3505` | `0.6495` | `0.1450` | `0.2998` | `0.1520` | `0.1777` |
| `boundary_34` | `0.7302` | `0.1497` | `0.0749` | `0.7967` | `0.2322` | `0.8316` | `0.0629` | `0.1875` | `0.1053` | `0.1512` |
| `boundary_34_refute` | `0.8369` | `0.2325` | `0.1525` | `0.6901` | `0.3683` | `0.5052` | `0.2663` | `0.3317` | `0.1544` | `0.1718` |
| `selective_v1` | `0.7992` | `0.1959` | `0.1265` | `0.7277` | `0.3323` | `0.6289` | `0.1926` | `0.2862` | `0.1264` | `0.1459` |
| `selective_v2` | `0.7955` | `0.2228` | `0.1442` | `0.7315` | `0.3495` | `0.6048` | `0.1934` | `0.3067` | `0.1474` | `0.1626` |
| `selective_v3` | `0.7309` | `0.2093` | `0.1107` | `0.7961` | `0.2697` | `0.7938` | `0.0721` | `0.2241` | `0.1290` | `0.1599` |
| `selective_v4` | `0.7848` | `0.2173` | `0.1391` | `0.7422` | `0.3445` | `0.6289` | `0.1750` | `0.2890` | `0.1353` | `0.1597` |

## 3. Main Conclusion

`boundary_34_selective_refute_v4` 比 `v3` 明显恢复到了更合理的区间，也比 `v1` 更平衡；  
但相对当前最稳的 `selective_v2`，它**没有形成明确提升**，更像是一个接近但略弱的变体。

## 4. What Improved vs V3

和 `selective_v3` 相比，v4 把 first-pass 从“明显偏 SAT”拉回来了：

- `F1-DSAT: 0.2697 -> 0.3445`
- `false_sat_rate: 0.7938 -> 0.6289`
- `pred SAT / DSAT: 1440 / 154 -> 1258 / 336`

这说明 v4 的“同时比较最强的 3/4 证据”确实比 v3 那种“普通缺口 vs 关键缺口”更稳，不会再明显滑向 SAT。

## 5. Comparison to V2

和 `selective_v2` 对比，v4 的位置可以概括成：

- 边界能力非常接近
- 但没有明显更好

具体看：

- `F1-DSAT: 0.3495 -> 0.3445`（略降）
- `false_sat_rate: 0.6048 -> 0.6289`（略差）
- `false_dsat_rate: 0.1934 -> 0.1750`（略好）
- `PU-bin F1-DSAT: 0.3067 -> 0.2890`（下降）
- `PU-bin Kappa: 0.1474 -> 0.1353`（下降）
- `WC-bin Pearson: 0.1626 -> 0.1597`（基本持平略降）

因此 v4 的效果更像：

- 用更低的 `false_dsat_rate`
- 换来了稍差的 DSAT 抓取能力和稍弱的用户级边界指标

它没有像 v3 那样明显失败，但也没有超过 v2。

## 6. First-Pass and Second-Pass Diagnostics

### 6.1 Trigger behavior

在 v4 中：

- `selective_refute_model_flag = 204`
- `selective_refute_triggered = 147`
- `trigger_rate = 9.2%`

对比：

- `v1`: `47.9%`
- `v2`: `11.6%`
- `v4`: `9.2%`

所以 v4 仍然保持了 selective 路线的低触发特征，没有重新把 second-pass 放大。

### 6.2 First-pass itself is already decent

v4 的 first-pass 指标：

- `F1-DSAT = 0.3385`
- `false_sat_rate = 0.6254`
- `false_dsat_rate = 0.1873`

final 指标：

- `F1-DSAT = 0.3445`
- `false_sat_rate = 0.6289`
- `false_dsat_rate = 0.1750`

这说明：

- first-pass 本身已经比较接近最终结果
- second-pass 仍然只是轻微修正，不是主导模块

### 6.3 Second-pass helps a bit more than in V2

v4 的 second-pass 不像 v2 那样几乎完全不动：

- 触发样本：`147`
- 真正改判：`19`
- `change rate | triggered = 12.9%`

改判模式：

- `(3 -> 3)`: `128`
- `(3 -> 4)`: `18`
- `(4 -> 3)`: `1`

改判净效果：

- `3 -> 4` 帮助真实 SAT：`16`
- `3 -> 4` 误伤真实 DSAT：`2`
- `4 -> 3` 帮助真实 DSAT：`1`
- `4 -> 3` 误伤真实 SAT：`0`

这说明 v4 的 second-pass 比 v2 稍微“活”一点，而且总体上改判质量是正向的。  
但它带来的收益仍然是有限的，无法把整体结果推到明显优于 v2。

## 7. Interpretation

v4 的意义在于，它验证了一件事：

- 如果 first-pass 采用更平衡的“双边证据比较”框架，确实可以避免再次滑向 SAT 或 DSAT 某一边

但它也说明：

- 这种平衡框架虽然稳，却不一定自动带来更强的个性化边界能力
- 在当前 Qwen3-8B + selective 系统下，`v2` 仍然是更好的折中点

换句话说：

- `v3` 过于“保护 4”
- `v1` 过于“怀疑 4”
- `v4` 更平衡
- 但综合指标上依然没有打败 `v2`

## 8. Practical Takeaway

当前 selective 版本的排序可以概括为：

- 最稳的主线：`selective_v2`
- 可接受但未超过主线：`selective_v4`
- 明显退化：`selective_v3`

所以如果你的目标是继续推进可用方案，而不是继续做 prompt 探索，  
目前仍应把 `boundary_34_selective_refute_v2` 作为主线版本。

## 9. Next Step Suggestion

如果继续在 selective 框架里优化，下一步不建议继续大改 prompt 的判定哲学。  
更值得做的是小幅、可控的局部增强，例如：

- 针对 `3 + 不够细致` 的样本做更细粒度的类型拆分
- 针对 `4 + 不满足需求` 的少量高风险样本加强复核
- 或直接利用当前输出中的 first-pass / follow-up 元信息做更轻量的后处理

也就是说，后续应从“大 prompt 形态变更”转向“小范围定点修补”。
