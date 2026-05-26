"""
PersTurnBench main benchmark figure.

Reads `paper-draft/Figures/persturnbench_main_refcdf_data.csv` and renders an
academic-quality 2-panel comparison of LLMs under reference-CDF calibrated
scoring:

  Left panel  : forest plot of user-macro satisfaction with 95% CI;
                each point is annotated with the mean value;
                a dashed vertical line marks the best model's mean.

  Right panel : stacked horizontal bars of the 1-5 calibrated score
                distribution per model (percent of replay turns), using a
                diverging palette around the 3 / 4 satisfaction boundary;
                DSAT (= scores 1-3) is annotated at the right margin.

Both panels share the model y-axis ordering (best → worst, top → bottom).

Run from `detection/`:
    python assets/plot_persturnbench_main_refcdf.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets/paper-draft/Figures/persturnbench_main_refcdf_data.csv"
OUT_DIR = ROOT / "assets/paper-draft/Figures"
OUT_STEM = "persturnbench_main_refcdf"


# ─── palette ────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    1: "#b22222",  # firebrick      (very dissatisfied)
    2: "#e07b39",  # warm orange    (dissatisfied)
    3: "#e8c547",  # amber          (neutral)
    4: "#7fb069",  # light green    (satisfied)
    5: "#2f7d3b",  # deep green     (very satisfied)
}
CI_DOT_COLOR = "#1f4e79"     # deep navy for forest dots
CI_BAR_COLOR = "#4a7ab4"     # softer navy for CI whiskers
BEST_REF_COLOR = "#888888"
DSAT_TEXT_COLOR = "#9b1c1c"


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    with open(DATA, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            rows.append(r)
    rows.sort(key=lambda r: float(r["user_macro"]), reverse=True)
    return rows


def main() -> None:
    rows = _load_rows()

    labels = [r["display_model"] for r in rows]
    user_macro = [float(r["user_macro"]) for r in rows]
    ci_lo = [float(r["user_macro_ci_low"]) for r in rows]
    ci_hi = [float(r["user_macro_ci_high"]) for r in rows]
    dsat = [100 * float(r["dsat_rate"]) for r in rows]

    score_pct = []  # list of (p1, p2, p3, p4, p5) per row
    for r in rows:
        counts = [float(r[f"score_{i}_count"]) for i in range(1, 6)]
        n = sum(counts) or 1.0
        score_pct.append([100 * c / n for c in counts])

    # ─── matplotlib style ────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 12,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#333333",
        "ytick.color": "#1a1a1a",
        "ytick.major.size": 0,
        "xtick.major.size": 3.5,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
    })

    n = len(rows)
    y = list(range(n))
    y_top_first = list(reversed(y))  # rank 1 on top

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(11.2, max(3.6, 0.62 * n + 1.4)),
        gridspec_kw={"width_ratios": [1.0, 1.2], "wspace": 0.10},
        sharey=True,
    )

    # ============================================================
    # LEFT — forest plot of user-macro ± 95% CI
    # ============================================================
    best = user_macro[0]

    ax_l.axvline(
        best, color=BEST_REF_COLOR, linestyle=(0, (4, 4)),
        linewidth=1.0, alpha=0.7, zorder=1,
    )

    for yi, mu, lo, hi in zip(y_top_first, user_macro, ci_lo, ci_hi):
        ax_l.hlines(yi, lo, hi, color=CI_BAR_COLOR, linewidth=1.8,
                    alpha=0.9, zorder=2)
        cap_h = 0.18
        for x in (lo, hi):
            ax_l.vlines(x, yi - cap_h, yi + cap_h,
                        color=CI_BAR_COLOR, linewidth=1.4, zorder=2)
        ax_l.scatter([mu], [yi], s=70, color=CI_DOT_COLOR,
                     edgecolor="white", linewidth=1.0, zorder=3)

    # value annotation to the right of the right whisker
    pad = 0.012
    for yi, mu, hi in zip(y_top_first, user_macro, ci_hi):
        ax_l.text(
            hi + pad, yi, f"{mu:.2f}",
            ha="left", va="center", fontsize=10.5,
            color="#1a1a1a",
        )

    # axis limits with a bit of slack for value labels
    lo_min = min(ci_lo)
    hi_max = max(ci_hi)
    span = hi_max - lo_min
    ax_l.set_xlim(lo_min - 0.10 * span, hi_max + 0.18 * span)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(list(reversed(labels)), fontsize=11.5)
    ax_l.set_ylim(-0.55, n - 0.45)
    ax_l.set_xlabel(
        "User-macro satisfaction (95% CI)",
        fontsize=12, labelpad=8,
    )

    ax_l.grid(axis="x", linestyle=":", linewidth=0.6,
              color="#bdbdbd", alpha=0.8, zorder=0)
    for side in ("top", "right"):
        ax_l.spines[side].set_visible(False)
    ax_l.spines["left"].set_visible(False)
    ax_l.tick_params(axis="y", which="both", left=False)

    ax_l.text(
        best, n - 0.10, f"best = {best:.2f}",
        ha="center", va="bottom",
        fontsize=9.5, color=BEST_REF_COLOR, fontstyle="italic",
    )

    # ============================================================
    # RIGHT — stacked score distribution (% of replay turns)
    # ============================================================
    bar_h = 0.62
    for yi, pcts in zip(y_top_first, score_pct):
        left = 0.0
        for s in (1, 2, 3, 4, 5):
            w = pcts[s - 1]
            ax_r.barh(
                yi, w, left=left, height=bar_h,
                color=SCORE_COLORS[s], edgecolor="white", linewidth=0.6,
                zorder=2,
            )
            if w >= 6.0:
                ax_r.text(
                    left + w / 2, yi, f"{w:.0f}%",
                    ha="center", va="center",
                    fontsize=9.5,
                    color="white" if s in (1, 2, 5) else "#1a1a1a",
                    fontweight="bold",
                )
            left += w

    # DSAT annotation on the far right
    ax_r.set_xlim(0, 116)
    for yi, d in zip(y_top_first, dsat):
        ax_r.text(
            102, yi, f"{d:.1f}%",
            ha="left", va="center",
            fontsize=10.5, color=DSAT_TEXT_COLOR, fontweight="bold",
        )
    ax_r.text(
        102, n - 0.10, "DSAT",
        ha="left", va="bottom",
        fontsize=10.5, color=DSAT_TEXT_COLOR, fontweight="bold",
    )

    ax_r.set_xticks([0, 25, 50, 75, 100])
    ax_r.set_xticklabels(["0", "25", "50", "75", "100"])
    ax_r.set_xlabel("Score distribution (% of replay turns)",
                    fontsize=12, labelpad=8)
    ax_r.grid(axis="x", linestyle=":", linewidth=0.6,
              color="#bdbdbd", alpha=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.tick_params(axis="y", which="both", left=False)

    # ─── legend on the bottom ───────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=SCORE_COLORS[s], edgecolor="white",
              label=f"score {s}")
        for s in (1, 2, 3, 4, 5)
    ]
    legend_handles.append(
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=CI_DOT_COLOR, markeredgecolor="white",
               markersize=8, label="user-macro mean"),
    )
    legend_handles.append(
        Line2D([0], [0], color=CI_BAR_COLOR, linewidth=1.8,
               label="95% CI"),
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=7,
        frameon=False,
        fontsize=10.5,
        handlelength=1.6,
        columnspacing=1.2,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out)
        print(f"saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
