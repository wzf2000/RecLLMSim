# Boundary 3/4 Selective Refute V3 Results on 20-User Subset

## 1. Setup

本轮分析对象：

- 方法：`boundary_34_selective_refute_v3`
- backbone：`Qwen/Qwen3-8B`
- memory update：`none`
- 文件：`detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v3_u20.jsonl`

子集规模：

- `20` 个用户
- `1594` 个 assistant turns
- `346` 个 block

真实分布：

- SAT (`>=4`) = `1303` (`81.7%`)
- DSAT (`<=3`) = `291` (`18.3%`)

本次对比仍然全部在相同 `1594` 个 `sample_id` 上重算。

## 2. Main Results

### 2.1 V3 Metrics

- `MAE = 0.7309`
- `Pearson = 0.2093`
- `QWK = 0.1107`
- `Boundary Accuracy = 0.7961`
- `F1-DSAT = 0.2697`
- `false_sat_rate = 0.7938`
- `false_dsat_rate = 0.0721`

用户感知二分类指标：

- `PU-bin F1-DSAT = 0.2241`
- `PU-bin Kappa = 0.1290`
- `WC-bin Pearson = 0.1599`

预测分布：

- gold: `SAT=1303`, `DSAT=291`
- pred: `SAT=1440`, `DSAT=154`

### 2.2 Same-Subset Comparison

| Method | MAE | Pearson | QWK | Acc | F1-DSAT | False SAT | False DSAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7077` | `0.2999` | `0.2807` | `0.7629` | `0.3505` | `0.6495` | `0.1450` | `0.2998` | `0.1520` | `0.1777` |
| `boundary_34` | `0.7302` | `0.1497` | `0.0749` | `0.7967` | `0.2322` | `0.8316` | `0.0629` | `0.1875` | `0.1053` | `0.1512` |
| `boundary_34_refute` | `0.8369` | `0.2325` | `0.1525` | `0.6901` | `0.3683` | `0.5052` | `0.2663` | `0.3317` | `0.1544` | `0.1718` |
| `selective_v1` | `0.7992` | `0.1959` | `0.1265` | `0.7277` | `0.3323` | `0.6289` | `0.1926` | `0.2862` | `0.1264` | `0.1459` |
| `selective_v2` | `0.7955` | `0.2228` | `0.1442` | `0.7315` | `0.3495` | `0.6048` | `0.1934` | `0.3067` | `0.1474` | `0.1626` |
| `selective_v3` | `0.7309` | `0.2093` | `0.1107` | `0.7961` | `0.2697` | `0.7938` | `0.0721` | `0.2241` | `0.1290` | `0.1599` |

## 3. Main Conclusion

`boundary_34_selective_refute_v3` **明显退化**，而且退化方向很清楚：

- 它把 first-pass 又重新推回了“默认判 SAT”的方向
- 本质上更像 `boundary_34` 的回归版，而不是 `v2` 的改进版

最直观的证据是：

- `pred SAT / DSAT = 1440 / 154`
- 相比真实分布 `1303 / 291`
- 它明显低估了 DSAT

这也体现在边界指标上：

- `F1-DSAT: 0.3495 -> 0.2697`（相对 `v2` 大幅下降）
- `false_sat_rate: 0.6048 -> 0.7938`（大幅恶化）
- `PU-bin F1-DSAT: 0.3067 -> 0.2241`

虽然它在全局误差上更好：

- `MAE: 0.7955 -> 0.7309`

但这个收益本质上是通过更保守地贴近 SAT 主分布换来的，不是你关心的个性化满意边界能力提升。

## 4. Why V3 Failed

### 4.1 The first-pass wording over-protected `4`

v3 的 first-pass 设计里，重点强化了：

- `不够细致` 默认更接近普通缺口
- 只有影响可用性才应判 `3`

这本来是为了避免把“普通缺口”错判成 DSAT。  
但在 Qwen3-8B 上，副作用更大：

- 模型把大量本应判 `3` 的样本也一起保护成了 `4`
- 导致 `false_sat_rate` 明显爆炸

### 4.2 The problem is in first-pass, not in the gate or follow-up

这次触发与改判数据非常能说明问题：

- `selective_refute_model_flag = 50` (`3.1%`)
- `selective_refute_triggered = 24` (`1.5%`)
- 真正改判只有 `2` 个

也就是说：

- gate 已经非常紧
- second-pass 基本不参与决策
- 整体结果几乎完全由 first-pass 决定

first-pass 与 final 基本一致：

- first-pass `F1-DSAT = 0.2729`
- final `F1-DSAT = 0.2697`

所以这次退化可以直接归因于：

- **v3 的 first-pass prompt 本身把边界判定做得过宽松了**

### 4.3 V3 collapses toward the same failure mode as `boundary_34`

和 `boundary_34` 对比很像：

- `boundary_34`: `pred SAT=1463`, `F1-DSAT=0.2322`, `false_sat=0.8316`
- `selective_v3`: `pred SAT=1440`, `F1-DSAT=0.2697`, `false_sat=0.7938`

所以 v3 并不是在 `v2` 基础上的小幅优化，而是把系统往一个旧失败模式拉回去了：

- 高 Accuracy
- 低 DSAT recall
- 大量把真实不满意误判成满意

## 5. Secondary Diagnostics

### 5.1 Trigger pattern

触发样本很少：

- `24 / 1594`

触发原因仍然全部是：

- `不够细致`

触发任务分布：

- `礼物准备`: `8`
- `旅行规划`: `7`
- `菜谱规划`: `6`
- `技能学习规划`: `3`

### 5.2 Follow-up has almost no effect

在 `24` 个触发样本里：

- `(3 -> 3)`: `22`
- `(3 -> 4)`: `2`

这两个改判中：

- `1` 个对真实 SAT 有帮助
- `1` 个误伤了真实 DSAT

因此 second-pass 在 v3 上几乎没有任何可观收益。

## 6. Interpretation

v3 的失败说明一件事：

- 仅仅把 first-pass 改写成“普通缺口 vs 关键缺口”的语言框架，不会自动提升边界能力
- 对 Qwen3-8B 来说，这类措辞反而很容易被理解成“优先保护 4”

换句话说，模型吸收了：

- “不要因为不够细致就给 3”

但没有足够稳定地吸收：

- “核心问题未回答 / 关键要求未满足时必须给 3”

这导致 first-pass 的判定重心失衡。

## 7. Practical Takeaway

当前三代 selective 版本里：

- `v1`: 太宽，第二遍干预过强
- `v2`: 最平衡，接近 `Qwen none`
- `v3`: 仅改 first-pass 后明显退化

因此当前最合理的结论是：

- `boundary_34_selective_refute_v2` 仍然是 selective 路线下最稳的版本
- `v3` 不值得继续扩到更大子集或全量

## 8. Next Step

下一步如果继续优化 first-pass，不建议再沿着“进一步保护普通缺口”这个方向走。  
更合理的是反过来加强：

- 对 `3` 的必要条件写得更硬
- 尤其是：
  - 核心问题是否真正回答
  - 关键要求是否真正满足
  - 何种缺失会明确导致“仍然不满意”

也就是说，后续 first-pass 改进应更偏：

- **加强 `3` 的判定锚点**

而不是继续弱化它。
