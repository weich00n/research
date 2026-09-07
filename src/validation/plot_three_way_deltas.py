"""Grouped bar chart: net delta E[intention] (t0->t12) by arm, TPB vs Direct vs
Recsys. Reads outputs/analysis/run_comparisons/three_way/arm_summary.csv
(produced by compare_three_methodologies.py) -- no new LLM calls, no new runs.

Usage: python src/validation/plot_three_way_deltas.py
Output: outputs/analysis/run_comparisons/three_way/delta_by_arm.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "..", "outputs", "analysis", "run_comparisons",
                         "three_way", "arm_summary.csv")
OUT_PATH = os.path.join(HERE, "..", "..", "outputs", "analysis", "run_comparisons",
                         "three_way", "delta_by_arm.png")

ARM_ORDER = [
    "C0_static", "C1_social_only",
    "C2_news_only", "C2_ambient_combined", "C2_context_only", "C2_context_headwind",
    "C3_news_only", "C3_ambient_combined", "C3_context_only", "C3_context_headwind",
]
ARM_LABELS = {
    "C0_static": "C0\nstatic", "C1_social_only": "C1\nsocial",
    "C2_news_only": "C2\nnews", "C2_ambient_combined": "C2\nambient",
    "C2_context_only": "C2\ncontext", "C2_context_headwind": "C2\nheadwind",
    "C3_news_only": "C3\nnews", "C3_ambient_combined": "C3\nambient",
    "C3_context_only": "C3\ncontext", "C3_context_headwind": "C3\nheadwind",
}
METHOD_ORDER = ["tpb", "direct", "recsys"]
METHOD_LABELS = {"tpb": "TPB", "direct": "Direct", "recsys": "Recsys"}
# Same palette as the slide deck artifact.
COLORS = {"tpb": "#B5563C", "direct": "#3B6E8F", "recsys": "#7A8B4A"}


def main():
    df = pd.read_csv(CSV_PATH)
    pivot = df.pivot(index="arm", columns="methodology", values="net_delta")
    pivot = pivot.reindex(ARM_ORDER)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    n_methods = len(METHOD_ORDER)
    bar_width = 0.26
    x = range(len(ARM_ORDER))

    for i, method in enumerate(METHOD_ORDER):
        offsets = [xi + (i - (n_methods - 1) / 2) * bar_width for xi in x]
        values = pivot[method].values
        bars = ax.bar(
            offsets, values, width=bar_width,
            label=METHOD_LABELS[method], color=COLORS[method],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        for xpos, v in zip(offsets, values):
            if pd.isna(v):
                continue
            va = "bottom" if v >= 0 else "top"
            offset_pts = 3 if v >= 0 else -3
            ax.annotate(
                f"{v:+.2f}", (xpos, v), textcoords="offset points",
                xytext=(0, offset_pts), ha="center", va=va,
                fontsize=7.5, color="#3A3A3A",
            )

    ax.axhline(0, color="#666666", linewidth=1, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=10)
    ax.set_ylabel("Net Δ E[intention]  (t0 → t12)", fontsize=11)
    ax.set_title(
        "Net change in mean fertility intention by arm, TPB vs Direct vs Recsys\n"
        "100 agents, 12 weeks, Qwen2.5-14B",
        fontsize=12, pad=14,
    )
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="lower left", ncol=1)
    ymin = min(0, pivot.min(numeric_only=True).min()) - 0.08
    ymax = max(0, pivot.max(numeric_only=True).max()) + 0.08
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor="white", bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
