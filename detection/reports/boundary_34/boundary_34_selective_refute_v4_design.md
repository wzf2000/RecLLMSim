# Boundary 3/4 Selective Refute V4 Design

## 1. Motivation

在当前 selective 路线中：

- `v2` 是最稳的版本
- `v3` 只改 first-pass 后明显退化，原因是 first-pass 过度保护了 `4`

因此下一版 first-pass 不能继续沿“弱化 3、保护普通缺口”的方向走，  
但也不能简单反过来强打 `3`，否则又可能回到 `refute_v1` 的过保守。

## 2. Design Goal

`boundary_34_selective_refute_v4` 的目标是：

- 继续只改 first-pass
- 不改 gate
- 不改 second-pass
- 让 first-pass 明确、对称地比较：
  - 最强的 `3` 证据
  - 最强的 `4` 证据

也就是说，这版不再强调“默认保护哪一边”，而是强制模型做一个平衡比较。

## 3. Core Idea

v4 的 first-pass 要求模型按顺序做三件事：

1. 写出最强的 `3` 证据
   - 核心问题是否未被回答
   - 关键要求是否被漏掉
   - 缺口是否已经明显影响可用性
2. 写出最强的 `4` 证据
   - 核心问题是否已经被回答
   - 关键要求是否已基本满足
   - 剩余问题是否只是普通缺口
3. 比较哪一边更强
   - `3` 证据更强则判 `3`
   - `4` 证据更强则判 `4`

只有在两边最强证据真的势均力敌时，才允许 `needs_refute_review=true`。

## 4. Why This May Work Better

和 v3 相比，v4 的预期优势是：

- 不再只强调“普通缺口不应掉到 3”
- 也不把 first-pass 重新变成 failure-first
- 而是让模型必须同时考虑两边最强证据

这有助于避免两种单边偏移：

- 偏 SAT：只看到“还算有帮助”的一面
- 偏 DSAT：只抓住某个缺口不放

## 5. Implementation

代码改动：

- `detection/lib/memory.py`
  - 新增 `boundary_34_selective_refute_v4` first-pass prompt
- `detection/trace/collect_personalized.py`
  - 将 `v4` 接到 selective 路由
  - gate 继续复用 `v2`
  - second-pass 继续复用 `boundary_34_selective_refute_v2_followup`
- shell scripts
  - 补充 `v4` 名称

因此，v4 是一个非常干净的实验：

- 只测试 first-pass 从“偏保护 4”改为“平衡比较两边证据”之后，会不会比 v2 更稳

## 6. Expected Outcome

理想情况下，v4 相比 v2 应满足：

- `false_sat_rate` 不明显恶化
- `false_dsat_rate` 不明显恶化
- `F1-DSAT` 至少保持接近 v2
- `PU-bin Kappa / WC-bin Pearson` 有小幅提升

更关键的是：

- 不能再像 v3 那样明显滑向 SAT
- 也不能回到 `refute_v1` 式的明显偏 DSAT

## 7. Suggested Run

建议先跑固定 20 用户子集，与 v2 直接对比：

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_selective_refute_v4 \
limit_users=20 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v4_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

如果 v4 的边界能力优于 v2，再考虑扩大到更大的用户子集。
