from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "outputs/static_replay"
ORIGINAL_FILE = ROOT / "outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl"
OUTPUT = ROOT / "assets/static_replay_vs_original_refcdf_wtl.png"

MODEL_FILES = {
    "kimi-k2.6": "kimi-k2.6_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl",
    "glm-5.1": "glm-5.1_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl",
    "deepseek-v4-pro": "deepseek-v4-pro_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl",
    "gpt-5.5": "gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl",
    "claude-opus-4-7": "claude-opus-4-7_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl",
    "minimax-m2.7": "minimax-m2.7_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(candidate_path: Path, original_by_id: dict[str, dict]) -> dict:
    rows = load_jsonl(candidate_path)
    improve = tie = worse = 0
    deltas: list[float] = []
    for row in rows:
        sid = row["sample_id"]
        if sid not in original_by_id:
            continue
        cand = float(row["pred_score"])
        orig = float(original_by_id[sid]["pred_score"])
        delta = cand - orig
        deltas.append(delta)
        if delta > 0:
            improve += 1
        elif delta < 0:
            worse += 1
        else:
            tie += 1
    n = improve + tie + worse
    return {
        "n": n,
        "improve": improve / n if n else 0.0,
        "tie": tie / n if n else 0.0,
        "worse": worse / n if n else 0.0,
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
    }


def main() -> None:
    original_rows = load_jsonl(ORIGINAL_FILE)
    original_by_id = {row["sample_id"]: row for row in original_rows}
    results = [
        (model, summarize(STATIC_DIR / filename, original_by_id))
        for model, filename in MODEL_FILES.items()
    ]
    results.sort(key=lambda item: (item[1]["mean_delta"], item[1]["improve"]), reverse=False)

    labels = [model for model, _ in results]
    improves = [100 * stats["improve"] for _, stats in results]
    ties = [100 * stats["tie"] for _, stats in results]
    worses = [100 * stats["worse"] for _, stats in results]
    deltas = [stats["mean_delta"] for _, stats in results]

    green = "#4f8f2f"
    orange = "#f2a91f"
    red = "#e60000"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 15,
        "hatch.linewidth": 1.6,
        "axes.linewidth": 1.2,
    })

    fig_h = max(4.0, 0.62 * len(results) + 1.8)
    fig, ax = plt.subplots(figsize=(12.9, fig_h))
    y = list(range(len(results)))

    ax.barh(y, improves, color=green, edgecolor="white", hatch="..", height=0.72)
    ax.barh(y, ties, left=improves, color=orange, edgecolor="black", hatch="---", height=0.72)
    ax.barh(
        y,
        worses,
        left=[a + b for a, b in zip(improves, ties)],
        color=red,
        edgecolor="white",
        hatch="|||",
        height=0.72,
    )

    for i, (a, b, c) in enumerate(zip(improves, ties, worses)):
        segments = [
            (a / 2, a, green, "white"),
            (a + b / 2, b, orange, "black"),
            (a + b + c / 2, c, red, "white"),
        ]
        for x, value, box_color, text_color in segments:
            if value < 4.5:
                continue
            ax.text(
                x,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": box_color,
                    "edgecolor": "none",
                    "alpha": 0.96,
                },
            )
        ax.text(
            101.0,
            i,
            f"+{deltas[i]:.2f}" if deltas[i] >= 0 else f"{deltas[i]:.2f}",
            ha="left",
            va="center",
            fontsize=13,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([f"{x}%" for x in [0, 20, 40, 60, 80, 100]], fontsize=15)
    ax.set_xlabel("Outcome vs original assistant turn (refCDF/calCDF scores)", fontsize=16, labelpad=10)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    legend = [
        Patch(facecolor=green, edgecolor="white", hatch="..", label="Improve"),
        Patch(facecolor=orange, edgecolor="black", hatch="---", label="Tie"),
        Patch(facecolor=red, edgecolor="white", hatch="|||", label="Worse"),
        Patch(facecolor="none", edgecolor="none", label="right: mean delta"),
    ]
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=15,
        handlelength=2.0,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {OUTPUT}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
