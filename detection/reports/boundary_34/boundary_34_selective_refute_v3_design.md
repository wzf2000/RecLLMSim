# Boundary 3/4 Selective Refute V3 Design

## 1. Why V3

在 `boundary_34_selective_refute_v2` 上，当前结论已经比较明确：

- 触发条件已经足够收紧
- 第二遍 follow-up 现在几乎不改判
- 整体性能基本由 first-pass 决定

因此下一步最合理的优化点不再是 second-pass，而是：

- 直接提升 first-pass 的 `3/4` 边界判断质量

## 2. Design Goal

v3 的目标是只改第一遍，不改第二遍与 gate：

- 保留 `boundary_34_selective_refute_v2` 的 trigger gate
- 保留 `boundary_34_selective_refute_v2_followup` 的第二遍复核
- 只优化 first-pass prompt 的判定语言

这样做的好处是：

- 更容易定位收益到底来自 first-pass 本身
- 不会把改动混进 gate 或 follow-up，便于 A/B 对比

## 3. First-Pass Changes

相比 v2，v3 的第一遍更明确强调下面几点：

### 3.1 从“优秀 vs 一般”切回“过线 vs 没过线”

第一遍不再隐含地拿回复和高质量答案比较，而是更直接问：

- 这个回复是否已经达到该用户愿意接受为“基本满意”的最低线？

### 3.2 显式区分两类缺口

v3 强制模型把缺口区分成：

- `普通缺口`
  - 还不够细
  - 还可更完整
  - 还可更个性化
  - 但不影响“基本满意”
  - 应判 `4`
- `关键缺口`
  - 核心问题没回答
  - 关键约束没满足
  - 缺失已明显影响可用性
  - 应判 `3`

### 3.3 针对当前主要错误模式做保护

根据前面的分析，当前需要特别防止两种误判：

1. 把“只是普通不够细致”的样本误判成 `3`
2. 把“语气像在帮忙，但其实没回答核心问题”的样本误判成 `4`

因此 v3 明确加入了两条保护：

- `不够细致` 默认更接近普通缺口，除非已经严重到不可用
- 友好语气或表面帮助性不能替代核心问题是否被回答

## 4. Implementation

新增 prompt 版本：

- `boundary_34_selective_refute_v3`

实现方式：

- `detection/lib/memory.py`
  - 新增 v3 的 first-pass prompt
- `detection/trace/collect_personalized.py`
  - 把 v3 接到与 v2 相同的 selective 路由
  - 复用 v2 的 trigger gate
  - 复用 `boundary_34_selective_refute_v2_followup`
- shell scripts
  - 补充新的 prompt 名称

关键点：

- v3 不引入新的 follow-up prompt
- v3 不更改 gate 条件
- v3 是一个“只测 first-pass 文案优化”的干净实验

## 5. Expected Behavior

理想情况下，v3 相比 v2 应该表现为：

- first-pass 本身更稳
- `false_dsat_rate` 下降
- `F1-DSAT` 至少不下降
- `PU-bin Kappa / WC-bin Pearson` 有小幅提升

如果 v3 有收益，那么可以明确归因于：

- first-pass 对“普通缺口 vs 关键缺口”的表述更清晰

而不是 second-pass 或 gate 带来的变化。

## 6. Suggested Run

建议仍然先在固定子集上做对比：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_selective_refute_v3 \
limit_users=20 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v3_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

若信号稳定，再扩到更大的用户子集。
