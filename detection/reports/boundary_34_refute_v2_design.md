# `boundary_34_refute_v2` Prompt 设计说明

## 目标

在 `boundary_34_refute` 的基础上，新增一个更温和的边界版本：

- 名称：`boundary_34_refute_v2`
- 输出仍然只允许 `3` 或 `4`
- 保留 refute / failure-first 框架
- 但显式收紧 `3` 的触发条件，减少对真实满意样本的误伤

## 为什么需要 v2

`boundary_34_refute` 的实验结果表明：

- 它成功降低了 `false_sat_rate`
- 也提高了 `F1-DSAT`
- 但代价是显著提高了 `false_dsat_rate`

也就是说，v1 的 refute 方向是对的，但判得太重，容易把：

- “有普通缺口，但仍达到最低满意线”

误判成：

- “未达满意线”

因此 v2 的核心不是放弃 refute，而是把“普通缺口”和“致命缺陷”明确分开。

## 设计思路

`boundary_34_refute_v2` 保留 failure-first，但把判 `3` 的门槛收紧为：

- 只有存在 **明确且关键的失败** 时，才判 `3`

如果回复满足以下条件，则应优先保护 `4`：

- 已回答核心问题
- 关键约束基本满足
- 剩余问题只是“不够细致 / 还可以更好”

## 关键改动

### 1. 明确区分两类问题

新版 prompt 强制模型区分：

- **致命缺陷**：足以掉到 `3`
- **普通缺口**：仍达到最低满意线，但不足以到 `5`

### 2. `不够细致` 不再默认等于 `3`

v1 中，Qwen 很容易把：

- 信息缺一点
- 细节不够满

直接解释成“没达最低满意线”。

v2 明确规定：

- `不够细致` 只有在严重到影响可用性、影响核心需求满足时，才允许判 `3`

### 3. 对 `4` 增加保护

新版 prompt 显式告诉模型：

- 如果核心问题已回答，关键要求也基本满足，应优先保护 `4`

这一步的目的是把分布从 `boundary_34_refute` 的偏保守状态往真实满意比例拉回一些。

### 4. 压短 `analysis`

v2 只要求 2-3 句简短分析，不再鼓励长篇反证链。

这既是为了减少保守倾向，也是为了降低生成长度和推理延迟。

## 预期效果

理想情况下，v2 应该相对 `boundary_34_refute` 呈现：

- `false_dsat_rate` 下降
- `F1-SAT` 回升
- `Accuracy` 回升
- 同时尽量保住：
  - `F1-DSAT`
  - `false_sat_rate`
  - `PU-F1-DSAT`

换句话说，目标不是继续把模型推向“更会抓不满意”，而是：

- **在保持 refute 优势的前提下，减少对 SAT 的误伤**

## 运行方式

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_refute_v2 \
bash scripts/collect_personalized_vllm.sh
```

输出文件会自动带上 `_boundary_34_refute_v2` 后缀。
