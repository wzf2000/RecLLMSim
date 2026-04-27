# Qwen3 Memory V2 Cache Analysis

## 1. Goal

本报告回答两个问题：

1. 当前 `Qwen3-8B` 的 `memory_v2` cache 总结是否合理
2. memory 对最终预测的影响是否合理，以及后续还有哪些可优化空间

分析对象：

- cache 目录：`detection/outputs/personalized/memory_cache/`
- 仅统计 `Qwen_Qwen3-8B` 的 `v2` memory 文件
- 结果文件对照：
  - `Qwen_Qwen3-8B_test_no_memory.jsonl`
  - `Qwen_Qwen3-8B_test_none.jsonl`
  - `Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2.jsonl`
  - `Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_fullscale.jsonl`

## 2. Cache-Level Summary

Qwen3 cache 文件总数：

- `356`

全部为：

- `memory_version = v2`

平均规模：

- `n_history_sessions = 12.3`
- `n_history_turns = 54.26`
- `user_specific_requirements` 平均 `3.58` 条
- `task_specific_observations` 平均 `3.0` 条

边界文本平均长度：

- `three_vs_four_distinction` 平均 `99.1` 字
- `four_vs_five_distinction` 平均 `126.3` 字

这说明当前 cache 在**信息量**上是足够的，不存在“summary 太短、明显缺字段”的问题。

## 3. Is the Summary Structurally Reasonable?

### 3.1 合理的部分

当前 cache 的几个方面是合理的：

- `avg_satisfaction_score` 和 `score_distribution` 都被稳定写出
- `three_vs_four_distinction` / `four_vs_five_distinction` 基本都是非空
- `user_specific_requirements` 大多不是完全重复模板
- `task_specific_observations` 能按任务保留不同描述

从抽样结果看，确实存在一些明显带用户特征的 memory，例如：

- 精确到地理位置、时间管理、预算区间
- 对特定菜系、礼物受众、学习资源形式的偏好

这说明 v2 相比 v1 的“完全泛化 summary”已经明显前进。

### 3.2 不太合理的部分

虽然结构齐全，但仍有几个明显问题：

#### 问题 A：`scoring_style` 几乎失去区分度

`scoring_style` 的统计：

- `严格`: `326`
- `偏严格`: `30`

也就是说，`356` 份 cache 里，这个字段几乎塌成两个值，而且绝大多数都是 `严格`。

这意味着：

- 这个字段对 Qwen 来说几乎不提供真实区分信息
- 真正有用的严格度信号其实来自 `avg_satisfaction_score`

#### 问题 B：`preferred_response_format` 也高度同质化

虽然 exact string 大多不同，但内容高度相似，主模式几乎都是：

- 结构清晰
- 分点列出
- 具体细节
- 实用建议

因此这个字段在当前形式下更像一个“低增益 filler”，而不是强判别特征。

#### 问题 C：`user_specific_requirements` 仍有相当比例偏泛

所有 requirements 总数：

- `1273`

粗略统计：

- 含“详细/清晰/具体/实用/完整”等泛化表述的比例：`0.431`
- 含品牌/价格/时间/步骤/链接/预算等更可操作线索的比例：`0.628`

这说明：

- 当前 requirements 不是完全泛化
- 但仍有大约四成带明显“通用好回答”色彩

因此这个字段属于：

- **有用，但纯度还不够高**

## 4. Evidence Coverage Problems

当前 cache 最大的结构性问题，不是“字段没写出来”，而是：

- **有些边界总结在证据不足时仍然被硬写出来**

### 4.1 Score coverage

按 `score_distribution` 统计：

- `score_5 == 0`: `13`
- `score_4 == 0`: `16`
- `score_3 == 0`: `99`
- `score_1 + score_2 == 0`: `193`
- `score_1 + score_2 + score_3 == 0`: `73`

相邻边界缺证据的 block 数量：

- `3/4` 缺直接相邻证据（`score_3==0` 或 `score_4==0`）：`103`
- `4/5` 缺直接相邻证据（`score_4==0` 或 `score_5==0`）：`29`

这说明：

- `3/4` 边界证据缺失是更严重的问题
- `1/2/3` 细分也天然更难，因为大量用户几乎没有低分历史

### 4.2 Hallucinated boundary risk

更关键的是，在证据缺失时，cache 并不总是显式保守。

观察到：

- `score_5 == 0` 但 `four_vs_five_distinction` 仍写出具体 4→5 规则的情况：存在
- `score_3 == 0` 但 `three_vs_four_distinction` 仍写出具体 3→4 规则的情况：很多

尤其是 `score_3 == 0` 的 block 中，大量 summary 仍然写出：

- “4分轮次可能因……降至3分”
- “3分轮次可能因……”

这类表述在没有相应真实历史时，本质上就是**推断性生成**，不是严格基于证据的总结。

这会带来两个风险：

1. router 被虚构的 `3/4` 规则过度牵引
2. DSAT 分支被放大，尤其在 boundary/fullscale 路线中更明显

## 5. Does Memory Affect Prediction in a Reasonable Way?

### 5.1 Raw `none` setting: yes, strongly and reasonably

以 block 为单位，比较 memory 的 `avg_satisfaction_score` 与 block 平均分：

- `corr(memory_avg, gold_block_mean) = 0.6649`

这说明 memory 中记录的“用户严格度”确实与真实 target block 分布强相关。

再看不同系统中，`memory_avg` 与预测 block 均值的相关性：

- `no_memory`: `0.1551`
- `none`: `0.5447`
- `boundary_v2`: `0.2567`
- `fullscale`: `0.2338`

这非常关键：

- 在原始 `Qwen none` 路线里，memory 明显改变了模型输出分布
- 而且这种改变方向是合理的，因为它显著接近真实用户均值结构

所以结论很明确：

**memory 对 Qwen3 的 raw 1-5 预测是有效的，而且主要是通过“用户级绝对刻度校准”在起作用。**

### 5.2 Boundary / Fullscale setting: effect is weaker and less well used

到了 `boundary_v2` 和 `fullscale`，上述相关性明显下降：

- `boundary_v2`: `0.2567`
- `fullscale`: `0.2338`

这说明：

- memory 仍然存在
- 但 router / boundary prompt 没有像 raw `none` 那样充分利用其中的校准信号

换句话说：

- raw `none` 更像在吸收 memory 的“用户严格度”
- boundary 系列更像在被 `three_vs_four_distinction` 的局部规则牵着走

这也解释了为什么：

- boundary / fullscale 更容易滑向 SAT 或 DSAT 偏置
- 而 raw `none` 在全局校准上反而更稳

## 6. Is the Current Memory Summary Reasonable Overall?

我的判断是：

**整体合理，但还不是最优形态。**

更准确地说：

- `memory_v2` 的大方向是对的
- 对 Qwen3 也确实产生了正向作用
- 但当前 cache 的真正强项是“提供用户级 calibration prior”
- 而不是“稳定提供高质量、严格证据化的边界规则”

因此如果把当前 cache 理解为：

- 一个高保真的用户评分 rubric

那这个理解有点过头。

如果把它理解为：

- 一个部分个性化、部分泛化，但对用户严格度很有帮助的混合先验

这个理解更接近事实。

## 7. Practical Optimization Directions

### 7.1 Highest priority: add evidence sufficiency flags

最该优先补的是：

- `has_score_5_evidence`
- `has_score_4_evidence`
- `has_score_3_evidence`
- `has_low_score_evidence`

以及更直接的：

- `can_compare_4_vs_5`
- `can_compare_3_vs_4`

然后在 prompt 中强制：

- 若缺相邻分数证据，不得把边界描述写成确定性规则
- 必须显式写成“证据不足，仅能弱推断”

这会直接减少当前 boundary/fullscale 中最危险的幻觉边界。

### 7.2 Reduce low-value textual fields

对 Qwen3 而言，当前这两个字段增益很低：

- `scoring_style`
- `preferred_response_format`

可以考虑：

- 压缩成更短模板
- 或在 boundary/fullscale prompt 中弱化它们

把 token 预算更多留给：

- `avg_satisfaction_score`
- `score_distribution`
- `three_vs_four_distinction`
- `four_vs_five_distinction`
- 真实案例级 evidence

### 7.3 Separate calibration memory from rule memory

当前 memory 里混合了两类信息：

1. calibration 信息
   - `avg_satisfaction_score`
   - `score_distribution`
   - 严格/宽松
2. rule / preference 信息
   - `three_vs_four_distinction`
   - `four_vs_five_distinction`
   - `user_specific_requirements`

对 Qwen3 来说，建议把这两类信息在 prompt 中分区得更明确：

- 先告诉模型“这个用户通常打分偏高/偏低到什么程度”
- 再告诉它“边界规则是什么”

这样更可能保留 raw `none` 的校准收益，同时减少 boundary prompt 的单边偏置。

### 7.4 For fullscale: branch-aware evidence gating

fullscale 当前的问题之一是：

- 4/5 分支会被“没有 5 分证据”的 memory 强行驱动
- 1/2/3 分支会被“几乎没有低分证据”的 memory 强行细化

所以 branch refine 应该感知证据充分性：

- 若 `score_5 == 0`，`4/5` 分支默认保守，避免强行推 `5`
- 若 `score_1 + score_2 + score_3` 很少，`1/2/3` 分支默认收缩到 `3`

这会比现在的统一细化更合理。

## 8. Bottom Line

当前 `Qwen3 memory_v2` cache 的结论可以压缩成一句话：

**它是有效的，但最有效的部分是“用户级分数校准”，而不是“严格证据化的边界规则总结”。**

因此后续优化 memory 时，不建议简单地继续增加字段或加长总结，而应该优先做：

1. 证据充分性显式化
2. 校准信息与边界规则分离
3. 弱化低增益通用文本字段
4. 根据可用分数证据决定 branch refinement 的保守程度
