"""Plot per-agent fertility-intention (and TPB) trajectories over time.

Each run JSON preserves every agent's weekly trajectory in `belief_history`
(t=0 frozen baseline .. t=N), so this script draws:

  1. per run   -> trajectories_<label>.png : spaghetti of E[intention] +
                  TPB small multiples (attitude / norm / pbc), one thin line
                  per agent, bold mean, flagged agents highlighted.
  2. cross-run -> trajectories_compare.png : mean +/- IQR band per run,
                  overlaid by label (only when >1 run is given).

It also flags "dubious" trajectories numerically (printed + CSV):

  - jump          : any one-step |dE[intention]| > 1.0
  - big_range     : max-min of E[intention] > 2.0 across the run
  - contradiction : net intention change (t0 -> tN, |d| > 0.25) opposite in
                    sign to ALL THREE net TPB deltas — the "LLM shortcutting
                    the scaffold" case CLAUDE.md calls scientifically
                    interesting; reported, not hidden
  - flatline      : zero variance in E[intention] across all timesteps
  - boundary      : E[intention] pinned < 1.2 or > 4.8 for >= 5 consecutive steps

Read-only. Reuses load_run / expected_intention from compare_runs.py and
mirrors its conventions (run from src/, outputs under outputs/analysis/).

Run from src/:
    python validation/plot_trajectories.py
    python validation/plot_trajectories.py \
        --runs ../outputs/runs/c2_llm_100.json ../outputs/runs/c2_cosine_100.json \
        --labels llm cosine --agents agent_007 agent_042
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this file lives in src/validation/; put src/ on the path so imports work
# the same way they do for driver.py (which is run from src/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.compare_runs import expected_intention, load_run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "outputs")  # src/validation/ -> repo root
RUNS = os.path.join(OUT, "runs")
ANALYSIS = os.path.join(OUT, "analysis", "trajectories")

DEFAULT_RUNS = ["c2_llm_100.json", "c2_hybrid_100.json", "c2_cosine_100.json"]
DEFAULT_LABELS = ["llm", "hybrid", "cosine"]

SERIES_COLS = {"E_intention": "E[intention]", "attitude": "Attitude",
               "norm": "Subjective norm", "pbc": "PBC"}

# flag thresholds (documented in the module docstring)
JUMP_STEP = 1.0
RANGE_MIN = 2.0
CONTRA_MIN_DI = 0.25
BOUNDARY_LO, BOUNDARY_HI, BOUNDARY_STEPS = 1.2, 4.8, 5
MAX_LABELLED = 8  # label at most this many flagged agents per figure

# ── palette (dataviz reference palette, light mode) ─────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# categorical slots in fixed order — run 1/2/3/... keeps its hue everywhere
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
          "#e34948", "#e87ba4", "#eb6834"]
FLAG = "#e34948"  # highlight hue for flagged agents in the per-run figures

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK2, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


# ── data ─────────────────────────────────────────────────────────────────────

def tidy(run):
    """agent_id x timestep -> attitude, norm, pbc, E_intention (long format)."""
    rows = []
    for aid in sorted(run):
        for e in run[aid]["belief_history"]:
            rows.append({
                "agent_id": aid,
                "timestep": e["timestep"],
                "attitude": e["attitude_score"],
                "norm": e["subjective_norm_score"],
                "pbc": e["pbc_score"],
                "E_intention": expected_intention(e.get("fertility_intention_dist")),
            })
    return pd.DataFrame(rows).sort_values(["agent_id", "timestep"])


def _boundary_stick(e):
    """True if E[intention] sits outside [BOUNDARY_LO, BOUNDARY_HI] for
    BOUNDARY_STEPS consecutive timesteps."""
    run_len = 0
    for v in e:
        run_len = run_len + 1 if (v < BOUNDARY_LO or v > BOUNDARY_HI) else 0
        if run_len >= BOUNDARY_STEPS:
            return True
    return False


def compute_flags(df, label):
    """One row per agent with trajectory stats + boolean flags."""
    rows = []
    for aid, g in df.groupby("agent_id"):
        g = g.sort_values("timestep")
        e = g["E_intention"].to_numpy()
        d_i = e[-1] - e[0]
        deltas = {c: g[c].iloc[-1] - g[c].iloc[0] for c in ("attitude", "norm", "pbc")}
        contradiction = (abs(d_i) > CONTRA_MIN_DI
                         and all(v * d_i < 0 for v in deltas.values()))
        rows.append({
            "run": label, "agent_id": aid,
            "E_t0": e[0], "E_tN": e[-1], "net_dE": d_i,
            "max_step_jump": float(np.abs(np.diff(e)).max()),
            "range_E": float(e.max() - e.min()),
            "std_E": float(e.std()),
            "net_d_attitude": deltas["attitude"],
            "net_d_norm": deltas["norm"],
            "net_d_pbc": deltas["pbc"],
            "flag_jump": bool(np.abs(np.diff(e)).max() > JUMP_STEP),
            "flag_big_range": bool(e.max() - e.min() > RANGE_MIN),
            "flag_contradiction": contradiction,
            "flag_flatline": bool(e.std() < 1e-9),
            "flag_boundary": _boundary_stick(e),
        })
    out = pd.DataFrame(rows)
    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    out["any_flag"] = out[flag_cols].any(axis=1)
    return out


# ── plotting ─────────────────────────────────────────────────────────────────

def _style_axis(ax, title, tmax):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlim(0, tmax)
    ax.set_ylim(1, 5)
    ax.set_xticks(range(0, tmax + 1, 2))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("week", fontsize=9)


def plot_run(df, flags, label, color, path):
    """2x2 spaghetti: E[intention] + the three TPB constructs, flags in red."""
    tmax = int(df["timestep"].max())
    flagged = flags.loc[flags["any_flag"], "agent_id"].tolist()
    # label only the most extreme flagged agents (selective direct labels)
    sev = flags.set_index("agent_id")[["range_E", "net_dE"]].abs().max(axis=1)
    labelled = set(sev.loc[flagged].sort_values(ascending=False)
                   .head(MAX_LABELLED).index)

    wide = {c: df.pivot(index="timestep", columns="agent_id", values=c)
            for c in SERIES_COLS}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for ax, (col, title) in zip(axes.flat, SERIES_COLS.items()):
        w = wide[col]
        for aid in w.columns:
            if aid in flagged:
                ax.plot(w.index, w[aid], color=FLAG, lw=1.4, alpha=0.85, zorder=3)
            else:
                ax.plot(w.index, w[aid], color=MUTED, lw=0.8, alpha=0.25, zorder=1)
        ax.plot(w.index, w.mean(axis=1), color=color, lw=2.2, zorder=4,
                solid_capstyle="round")
        _style_axis(ax, title, tmax)
        if col == "E_intention":  # agent_id labels once, on the headline panel
            ends = sorted(((float(w[a].iloc[-1]), a) for a in labelled))
            y_prev = -1.0
            for y, aid in ends:  # nudge collided end-labels apart
                y = max(y, y_prev + 0.16)
                y_prev = y
                ax.annotate(aid.replace("agent_", "a"), xy=(tmax, w[aid].iloc[-1]),
                            xytext=(tmax + 0.25, y), fontsize=7, color=INK2,
                            va="center", annotation_clip=False)

    handles = [plt.Line2D([], [], color=MUTED, lw=0.8, alpha=0.5),
               plt.Line2D([], [], color=FLAG, lw=1.4),
               plt.Line2D([], [], color=color, lw=2.2)]
    fig.legend(handles, ["agent", f"flagged agent (n={len(flagged)})", "mean"],
               loc="upper right", ncol=3, fontsize=9)
    fig.suptitle(f"Belief trajectories — {label} (n={df['agent_id'].nunique()} agents)",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_compare(dfs, labels, path):
    """Mean +/- IQR band of each series per run, overlaid by label."""
    tmax = int(max(df["timestep"].max() for df in dfs))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for ax, (col, title) in zip(axes.flat, SERIES_COLS.items()):
        for df, label, color in zip(dfs, labels, SERIES):
            g = df.groupby("timestep")[col]
            t = g.mean().index
            ax.fill_between(t, g.quantile(0.25), g.quantile(0.75),
                            color=color, alpha=0.10, lw=0)
            ax.plot(t, g.mean(), color=color, lw=2.0, label=label,
                    solid_capstyle="round")
        _style_axis(ax, title, tmax)
    handles, leg_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right", ncol=len(labels), fontsize=9)
    fig.suptitle("Mean trajectory ± IQR by relevance mode",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


# ── report ───────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+",
                    default=[os.path.join(RUNS, f) for f in DEFAULT_RUNS])
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--agents", nargs="+", default=None,
                    help="restrict to these agent_ids (zoom into flagged agents)")
    ap.add_argument("--csv", default=os.path.join(ANALYSIS, "trajectory_flags.csv"))
    args = ap.parse_args()
    if len(args.runs) != len(args.labels):
        ap.error(f"--runs ({len(args.runs)}) and --labels ({len(args.labels)}) "
                 "must have the same length")

    os.makedirs(ANALYSIS, exist_ok=True)
    dfs, all_flags = [], []
    for path, label in zip(args.runs, args.labels):
        df = tidy(load_run(path))
        if args.agents:
            df = df[df["agent_id"].isin(args.agents)]
        dfs.append(df)
        all_flags.append(compute_flags(df, label))

    # t=0 must be identical across runs (shared frozen baseline) — free check
    if len(dfs) > 1:
        base = [df[df["timestep"] == 0].set_index("agent_id")[list(SERIES_COLS)]
                for df in dfs]
        worst = max(float((b - base[0]).abs().max().max()) for b in base[1:])
        status = "OK" if worst < 1e-9 else "MISMATCH — runs don't share the frozen baseline!"
        print(f"t=0 baseline identical across runs: max |diff| = {worst:.2e}  [{status}]\n")

    for df, flags, label, color in zip(dfs, all_flags, args.labels, SERIES):
        print(f"== {label} ==  agents: {df['agent_id'].nunique()}, "
              f"timesteps: {df['timestep'].min()}..{df['timestep'].max()}, "
              f"mean E[intention] t0 -> tN: "
              f"{df[df.timestep == 0]['E_intention'].mean():.3f} -> "
              f"{df[df.timestep == df.timestep.max()]['E_intention'].mean():.3f}")
        for col in [c for c in flags.columns if c.startswith("flag_")] + ["any_flag"]:
            ids = flags.loc[flags[col], "agent_id"].tolist()
            print(f"  {col:<20} {len(ids):>3}  {', '.join(ids) if ids else '-'}")
        # ratchet check: constructs that only ever go up / end pinned at the ceiling
        for col in ("attitude", "norm", "pbc"):
            w = df.pivot(index="timestep", columns="agent_id", values=col)
            mono = float((w.diff().iloc[1:] >= 0).all(axis=0).mean())
            ceil = float((w.iloc[-1] >= 4.8).mean())
            print(f"  ratchet {col:<9} monotone non-decreasing: {mono:5.0%}   "
                  f">=4.8 at t=max: {ceil:5.0%}")
        out_png = os.path.join(ANALYSIS, f"trajectories_{label}.png")
        plot_run(df, flags, label, color, out_png)
        print(f"  saved -> {out_png}\n")

    if len(dfs) > 1:
        cmp_png = os.path.join(ANALYSIS, "trajectories_compare.png")
        plot_compare(dfs, args.labels, cmp_png)
        print(f"saved -> {cmp_png}")

    pd.concat(all_flags, ignore_index=True).to_csv(args.csv, index=False,
                                                   encoding="utf-8")
    print(f"saved -> {args.csv}")


if __name__ == "__main__":
    main()
