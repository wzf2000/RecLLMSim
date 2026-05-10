from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/static_replay/pairwise_vs_gpt55_refCDF.json"
OUTPUT = ROOT / "assets/static_replay_pairwise_vs_gpt55_refcdf_wtl.png"


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda r: r["micro"]["non_tie_win_rate"])

    labels = [r["candidate"] for r in rows]
    wins = [100 * r["micro"]["win_rate"] for r in rows]
    ties = [100 * r["micro"]["tie_rate"] for r in rows]
    losses = [100 * r["micro"]["lose_rate"] for r in rows]

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

    fig_h = max(3.8, 0.62 * len(rows) + 1.7)
    fig, ax = plt.subplots(figsize=(12.8, fig_h))
    y = list(range(len(rows)))

    ax.barh(y, wins, color=green, edgecolor="white", hatch="..", height=0.72)
    ax.barh(y, ties, left=wins, color=orange, edgecolor="black", hatch="---", height=0.72)
    ax.barh(
        y,
        losses,
        left=[w + t for w, t in zip(wins, ties)],
        color=red,
        edgecolor="white",
        hatch="|||",
        height=0.72,
    )

    for i, (w, t, l) in enumerate(zip(wins, ties, losses)):
        segments = [(w / 2, w), (w + t / 2, t), (w + t + l / 2, l)]
        for x, value in segments:
            if value < 4.5:
                continue
            if x < w:
                text_color = "white"
                box_color = green
            elif x <= w + t:
                text_color = "black"
                box_color = orange
            else:
                text_color = "white"
                box_color = red
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

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([f"{x}%" for x in [0, 20, 40, 60, 80, 100]], fontsize=15)
    ax.set_xlabel("Pairwise outcome vs GPT-5.5 (refCDF calibrated)", fontsize=16, labelpad=10)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    legend = [
        Patch(facecolor=green, edgecolor="white", hatch="..", label="Win"),
        Patch(facecolor=orange, edgecolor="black", hatch="---", label="Tie"),
        Patch(facecolor=red, edgecolor="white", hatch="|||", label="Lose"),
    ]
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=15,
        handlelength=2.0,
        columnspacing=2.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {OUTPUT}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
