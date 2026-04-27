# Boundary 3/4 Selective Refute V2 Design

## 1. Motivation

在 20 用户子集上，`boundary_34_selective_refute` 的问题主要有两点：

- 第一遍触发率过高
  - `764 / 1594 = 47.9%`
  - 说明很多并不真正接近 `3/4` 边界的样本也被送进了第二遍
- 第二遍整体偏向保护 SAT
  - first-pass `F1-DSAT = 0.3733`
  - final `F1-DSAT = 0.3323`
  - 说明 follow-up 把不少本来能抓到的不满意样本又拉回了满意

因此 v2 的目标不是推翻 selective 思路，而是：

1. 收紧第一遍的触发条件
2. 让第二遍从“偏向保护 4”改为“默认维持初判，只有明确反证才改判”

## 2. Design Changes

### 2.1 First Pass: Tighter Trigger

新增版本：`boundary_34_selective_refute_v2`

第一遍 prompt 仍输出：

- `classification`
- `reason`
- `analysis`
- `needs_refute_review`

但触发语义更严格：

- 只有在“唯一可疑点是否属于关键失败”拿不准时，才允许 `needs_refute_review=true`
- 不再因为泛泛的“不够细致”自动触发
- 不再因为“还可以更好”触发
- 强调明显 SAT / 明显 DSAT 都必须直接结束，不进入第二遍

### 2.2 Code-Level Trigger Filter

除了 prompt 约束，代码层再加一道收紧过滤：

- 若第一遍判 `3`
  - 只有 `reason in {"不够细致", "其它"}` 才允许真正进入第二遍
  - 如果第一遍已经给出 `不满足需求 / 不可用`，视为更明确的 DSAT，不再复核
- 若第一遍判 `4`
  - 只有 `reason in {"不满足需求", "其它", "不可用"}` 才允许进入第二遍
  - `4 + 不够细致` 不再自动复核

因此在输出记录里会出现两个字段：

- `selective_refute_model_flag`
  - 模型在第一遍里是否建议复核
- `selective_refute_triggered`
  - 经过代码层过滤后，是否真的进入第二遍

这样后续可以区分：

- 是模型本身触发太多
- 还是代码过滤后仍触发过多

### 2.3 Second Pass: Default Keep First Pass

第二遍 prompt 改成更保守的“核实模式”：

- 默认维持第一遍初判
- 只有发现【明确反证】时，才允许改判

具体规则：

- 若第一遍判 `3`
  - 只有当第二遍能明确指出“核心问题已被回答、关键约束也已满足”时，才可改为 `4`
- 若第一遍判 `4`
  - 只有当第二遍能明确指出“关键要求被漏掉、核心问题未被回答、或明显低于最低满意线”时，才可改为 `3`

禁止模糊改判：

- 不能因为“也许够了”把 `3` 改成 `4`
- 不能因为“还可以更好”把 `4` 改成 `3`

## 3. Implementation

代码改动：

- `detection/lib/memory.py`
  - 新增 `boundary_34_selective_refute_v2` 第一遍 prompt
  - 扩展 `build_turn_eval_refute_followup_prompt(..., prompt_version=...)`
  - 为 v2 提供新的 follow-up prompt
- `detection/trace/collect_personalized.py`
  - `_call_predict_turn()` 增加 v2 路由
  - 新增 `_should_trigger_selective_refute()`
  - 对 v2 启用代码层触发过滤
  - 第二遍 follow-up 版本改为 `boundary_34_selective_refute_v2_followup`
- 脚本入口
  - `collect_personalized.sh`
  - `collect_personalized_vllm.sh`
  - 均补充新 prompt 名称

## 4. Expected Effect

理想情况下，v2 应该同时出现下面三个现象：

1. `selective_refute_triggered` 比 v1 明显下降
2. second-pass 改判率下降，但改判质量更高
3. 最终结果相对 v1：
   - `false_sat_rate` 不要明显恶化
   - `false_dsat_rate` 下降
   - `F1-DSAT` 不要被第二遍过度拉低

最关键的判断标准不是单看 `Accuracy`，而是：

- v2 的 final 结果是否比 v1 更接近 first-pass 的 DSAT 抓取能力
- 同时又不像 `refute_v1` 那样明显过度保守

## 5. Suggested Run

先在小子集上跑：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_selective_refute_v2 \
limit_users=10 \
bash scripts/collect_personalized_vllm.sh
```

若结果稳定，再扩大到：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_selective_refute_v2 \
limit_users=20 \
bash scripts/collect_personalized_vllm.sh
```
