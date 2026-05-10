"""Turn-level evaluation prompt builders for user memory."""

from __future__ import annotations

from typing import Literal

from .satisfaction_constants import SATISFIED_REASON, get_reason_to_id
from .memory_schema import UserMemory
from .memory_formatting import (
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
)

# ──────────────────────────────────────────────────────────────────────────────
# Turn Evaluation Prompt（v2：rubric 式逐步判断）
# ──────────────────────────────────────────────────────────────────────────────

from .memory_eval_common import _format_anchor_turns


from .memory_eval_base_prompts import build_turn_eval_base_prompt
from .memory_eval_boundary_prompts import build_turn_eval_boundary_prompt
from .memory_eval_history_prior_prompts import (
    build_turn_eval_history_prior_episodic_refine_prompt,
    build_turn_eval_history_prior_prompt,
)
from .memory_eval_qwen_prompts import build_turn_eval_qwen_short_prompt
from .memory_eval_v3_prompts import build_turn_eval_v3_prompt
def build_turn_eval_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    anchor_turns: list | None = None,
    prompt_version: Literal[
        "v2", "v3", "v3_1", "history_prior_delta", "history_prior_delta_v2", "history_prior_delta_v3", "history_prior_delta_v3_1", "history_prior_delta_v3_episodic", "history_prior_delta_v3_episodic_twopass", "qwen_short", "boundary_34", "boundary_34_refute", "boundary_34_refute_v2",
        "boundary_34_selective_refute", "boundary_34_selective_refute_v2",
        "boundary_34_selective_refute_v3", "boundary_34_selective_refute_v4",
    ] = "v2",
) -> str:
    """
    构造单轮满意度预测 prompt（v2）。

    核心改进：将 memory 转化为逐步判断的个性化评分 rubric，
    而非泛化的"参考以下模式"。评分逻辑显式分三步：
      Step 1: 是否达到 4 分门槛（three_vs_four_distinction）
      Step 2: 若达到，是否进一步达到 5 分（four_vs_five_distinction）
      Step 3: 若未达到 4 分，根据缺陷程度判断 1/2/3 分

    若提供 anchor_turns（list[AnchorTurn]），会在 rubric 之后插入"参考案例"块，
    作为 few-shot in-context 锚点。

    prompt_version:
      - "v2": 保持原有 rubric prompt，不改历史实验行为
      - "v3": 分离 calibration 与 boundary 规则；证据不足时弱化边界总结
      - "v3_1": 在 v3 基础上重新加硬 3/4 最低满意线；证据不足只影响 1/2/3 细分，不放松 SAT gate
      - "history_prior_delta": 显式以历史均分为 prior，先判 residual delta 与 3/4 boundary，再重建最终分
      - "history_prior_delta_v2": soft reconstruction 版本，新增 confidence 与 strong evidence 字段
      - "history_prior_delta_v3": hybrid vote 版本，用多个 DSAT 信号触发降到 3，同时保持 prior exact-score anchor
      - "history_prior_delta_v3_1": v3 收紧版，仅在三个 DSAT 信号同时成立时触发降到 3
      - "history_prior_delta_v3_episodic": v3 + 边界成对 episodic anchors，用历史真实轮次辅助判 3/4
      - "history_prior_delta_v3_episodic_twopass": v3.1 first-pass + 不确定样本 episodic 二次复核
      - "qwen_short": 面向 Qwen3-8B 的更短、更硬的 checklist prompt
      - "boundary_34": 仅围绕 3/4 满意边界判断，输出限制为 3 或 4
      - "boundary_34_refute": 在 3/4 边界上先做反证检查，抑制默认判 4
      - "boundary_34_refute_v2": 更温和的 refute 版本，仅在存在明确致命缺陷时判 3
      - "boundary_34_selective_refute": 第一遍温和判 3/4，并显式标记是否需要二次反证复核
      - "boundary_34_selective_refute_v2": selective 的收紧版本，只在高不确定边界样本上触发二判
      - "boundary_34_selective_refute_v3": 仅优化 first-pass 的边界措辞，gate 和二判保持 v2
      - "boundary_34_selective_refute_v4": 平衡 first-pass，强制同时考虑最强的 3/4 证据
    """
    if prompt_version == "history_prior_delta_v3_episodic_twopass":
        return build_turn_eval_history_prior_prompt(
            memory, profile, task_context, history_window, assistant_reply, None, "history_prior_delta_v3_1"
        )
    if prompt_version in {"history_prior_delta", "history_prior_delta_v2", "history_prior_delta_v3", "history_prior_delta_v3_1", "history_prior_delta_v3_episodic"}:
        return build_turn_eval_history_prior_prompt(
            memory, profile, task_context, history_window, assistant_reply, anchor_turns, prompt_version
        )
    if prompt_version in {"v3", "v3_1"}:
        return build_turn_eval_v3_prompt(
            memory, profile, task_context, history_window, assistant_reply, anchor_turns, prompt_version
        )
    if prompt_version in {
        "boundary_34",
        "boundary_34_refute",
        "boundary_34_refute_v2",
        "boundary_34_selective_refute",
        "boundary_34_selective_refute_v2",
        "boundary_34_selective_refute_v3",
        "boundary_34_selective_refute_v4",
    }:
        return build_turn_eval_boundary_prompt(
            memory, profile, task_context, history_window, assistant_reply, anchor_turns, prompt_version
        )
    if prompt_version == "qwen_short":
        return build_turn_eval_qwen_short_prompt(
            memory, profile, task_context, history_window, assistant_reply, anchor_turns, prompt_version
        )
    return build_turn_eval_base_prompt(
        memory, profile, task_context, history_window, assistant_reply, anchor_turns, prompt_version
    )


from .memory_eval_refute_prompts import build_turn_eval_refute_followup_prompt
from .memory_eval_refinement_prompts import (
    build_turn_eval_fullscale_dsat_refinement_prompt,
    build_turn_eval_fullscale_sat_refinement_prompt,
    build_turn_eval_v3_two_stage_dsat_refinement_prompt,
    build_turn_eval_v3_two_stage_sat_refinement_prompt,
)
from .memory_eval_two_stage_prompts import (
    build_turn_eval_v3_two_stage_gate_prompt,
    build_turn_eval_v3_two_stage_v2_gate_followup_prompt,
    build_turn_eval_v3_two_stage_v2_gate_prompt,
)
from .memory_eval_no_memory_prompts import build_turn_eval_prompt_no_memory
