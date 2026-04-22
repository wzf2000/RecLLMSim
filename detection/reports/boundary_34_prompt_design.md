# `boundary_34` Prompt 设计说明

## 目标

在现有 `v2` / `qwen_short` 之外，新增一个**只面向 3/4 满意边界**的 turn eval prompt 版本。

该版本不试图完成完整的 1-5 分预测，而是把问题收缩为：

- `4` = 满意（达到该用户的最低满意线）
- `3` = 不满意（未达到该用户的最低满意线）

也就是说，输出仍兼容现有 `classification` 字段，但只允许取 `3` 或 `4`。

## 为什么这样做

基于 `reports/boundary_metrics_results.md` 的补充验证，目前项目真正更关心的是：

- 用户是否满意，而不是回复是否“足够优秀”
- 如何减少把真实不满意误判成满意（`false_sat_rate`）
- 如何建模每个用户的 **3/4 边界**

此前的 `qwen_short` 虽然改善了全局误差，但本质上是在牺牲 DSAT 边界识别，不能作为后续主线。

## 设计原则

`boundary_34` 版本做了三件事：

1. **目标收缩**
   不再让模型区分 `1/2/3/4/5`，只问：
   - 有没有达到最低满意线？

2. **memory 重心前移到 `three_vs_four_distinction`**
   `four_vs_five_distinction` 只保留为背景，不再是主要判断依据。

3. **输出硬约束**
   `classification` 只能是 `3` 或 `4`，避免模型把容量花在 4/5 或 1/2 的细分上。

## 预期用途

- 用于验证：把任务显式收缩到 3/4 边界之后，Qwen3-8B 是否能提升：
  - `F1-DSAT`
  - `false_sat_rate`
  - `PU-F1-DSAT`
  - `PU-Kappa`
  - `WC-bin-Pearson`

- 不建议将其与 1-5 分主指标直接比较；应优先用边界指标评估。

## 运行方式

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34 \
bash scripts/collect_personalized_vllm.sh
```

输出文件会自动带上 `_boundary_34` 后缀。
