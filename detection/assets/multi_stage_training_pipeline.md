# Satisfaction 多阶段训练流程图

## 汇报版精简图

```mermaid
flowchart LR
    A["原始对话数据<br/>用户画像、任务背景、历史对话、满意度标注"] --> B["阶段1 大模型采样<br/>生成初始推理轨迹与 JSON 预测"]
    B --> C["阶段2 反思增强<br/>对首轮错误样本生成 reflection 轨迹"]
    C --> D["轨迹池构建<br/>正确轨迹 + 反思纠错轨迹"]
    D --> E["阶段3 SFT 小模型<br/>学习评分、原因与可选推理内容"]
    E --> F["SFT 模型 checkpoint"]
    F --> G["阶段4 GRPO<br/>基于可验证奖励继续优化"]
    G --> H["最终小模型"]

    D -. 监督信号 .-> E
    D -. gold 标签 .-> G
    F -. 初始化策略 .-> G
```

## 奖励设计

```mermaid
flowchart TD
    A["GRPO 采样输出"] --> B["格式奖励<br/>JSON 可解析、标签合法"]
    A --> C["分数奖励<br/>预测分数接近 gold"]
    A --> D["原因奖励<br/>预测原因命中 gold"]
    B --> E["加权合成总奖励"]
    C --> E
    D --> E
    E --> F["更新策略模型"]
```

## 对应脚本

- `detection/collect_api_model_traces.py`：大模型采样与反思轨迹生成
- `detection/sft_from_traces.py`：基于轨迹做监督微调
- `detection/grpo_from_sft.py`：基于 SFT checkpoint 做 GRPO
- `detection/eval_sft_from_traces.py`：统一评测 SFT / GRPO 后模型

