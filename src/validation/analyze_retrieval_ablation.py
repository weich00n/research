"""Retrieval-mechanism ablation analysis: does the TPB<->intention mediation
result survive when memory retrieval is construct-blind (--retrieval saliency,
see sandbox/lesson.py:retrieve_memories_saliency_only)?

For each of the 4 ablation runs (run_C0/C1/C2/C3_saliencyretr.json, produced
by run_saliency_ablation.sh) this compares against its canonical counterpart
-- the same condition run with the CLAUDE.md formula (saliency x per-construct
TPB-relevance) -- on two things:

  1. SPECIFICITY: net per-agent delta (E[intention] at t0 vs tN, and each TPB
     construct at t0 vs tN). Does the hypothesised construct still move most
     under construct-blind retrieval, same as under the canonical formula?
  2. MEDIATION: pooled week-to-week correlation between each construct's
     delta and E[intention]'s delta (every agent x every consecutive-week
     transition = one data point). If saliency-only retrieval collapses this
     correlation relative to canonical, the mediation result depended on
     retrieval pre-filtering by relevance, not on the belief content itself.

Read-only, no new LLM calls. Run from src/:
    python validation/analyze_retrieval_ablation.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
RUNS = os.path.join(REPO, "outputs", "runs")
SMOKE = os.path.join(RUNS, "smoke")
OUT_DIR = os.path.join(REPO, "outputs", "analysis", "retrieval_ablation")

CONSTRUCTS = ["attitude_score", "subjective_norm_score", "pbc_score"]
CONSTRUCT_LABELS = {"attitude_score": "attitude", "subjective_norm_score": "norm", "pbc_score": "pbc"}

# (arm, saliency-ablation run name, canonical (relevance-formula) run name,
#  canonical run dir override or None -> defaults to RUNS)
ARMS = [
    ("C0_static",       "run_C0_saliencyretr", "c0_smoke_gated",    SMOKE),
    ("C1_social_only",  "run_C1_saliencyretr", "run_C1_Qwen_fixed", None),
    ("C2_combined",     "run_C2_saliencyretr", "run_C2_ambient",    None),
    ("C3_combined",     "run_C3_saliencyretr", "run_C3_ambient",    None),
]


def load_run(path):
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    return state["agents"]


def expected_intention(dist):
    if not dist:
        return float("nan")
    return float(sum((i + 1) * p for i, p in enumerate(dist)))


def trajectories(agents):
    """{agent_id: DataFrame(timestep, attitude_score, subjective_norm_score, pbc_score, E)}"""
    out = {}
    for a in agents:
        hist = sorted(a.get("belief_history") or [], key=lambda h: h["timestep"])
        if not hist:
            continue
        rows = [{"timestep": h["timestep"], **{c: h[c] for c in CONSTRUCTS},
                 "E": expected_intention(h.get("fertility_intention_dist"))} for h in hist]
        out[a["agent_id"]] = pd.DataFrame(rows)
    return out


def net_deltas(traj):
    """Per-agent (t0 -> tN) delta for E and each construct. DataFrame indexed by agent_id."""
    rows = []
    for aid, df in traj.items():
        t0, tN = df.iloc[0], df.iloc[-1]
        row = {"agent_id": aid, "dE": tN["E"] - t0["E"]}
        for c in CONSTRUCTS:
            row[f"d_{CONSTRUCT_LABELS[c]}"] = tN[c] - t0[c]
        rows.append(row)
    return pd.DataFrame(rows)


def weekly_deltas(traj):
    """Pooled week-to-week deltas across all agents (every consecutive-timestep
    transition is one row) -- the mediation correlation's input."""
    rows = []
    for df in traj.values():
        d = df.sort_values("timestep").diff().dropna()
        for c in CONSTRUCTS:
            rows.append(d.rename(columns={c: "d_construct"})[["d_construct"]]
                        .assign(construct=CONSTRUCT_LABELS[c], dE=d["E"]))
    if not rows:
        return pd.DataFrame(columns=["construct", "d_construct", "dE"])
    return pd.concat(rows, ignore_index=True)


def corr_stats(x, y):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return {"n": len(x), "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan")}
    r, p = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y).correlation
    return {"n": len(x), "pearson_r": float(r), "pearson_p": float(p), "spearman_r": float(sr)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    specificity_rows, mediation_rows = [], []

    for arm, sal_name, canon_name, canon_dir in ARMS:
        sal_path = os.path.join(RUNS, f"{sal_name}.json")
        canon_path = os.path.join(canon_dir or RUNS, f"{canon_name}.json")

        for method, path, run_name in [("saliency_blind", sal_path, sal_name),
                                        ("canonical_relevance", canon_path, canon_name)]:
            if not os.path.exists(path):
                print(f"WARNING: missing {path} ({arm}/{method}) -- skipping")
                continue
            agents = load_run(path)
            traj = trajectories(agents)

            nd = net_deltas(traj)
            specificity_rows.append({
                "arm": arm, "method": method, "run": run_name, "n_agents": len(nd),
                "mean_dE": nd["dE"].mean(),
                **{f"mean_d_{CONSTRUCT_LABELS[c]}": nd[f"d_{CONSTRUCT_LABELS[c]}"].mean() for c in CONSTRUCTS},
            })

            wd = weekly_deltas(traj)
            for construct in CONSTRUCT_LABELS.values():
                sub = wd[wd["construct"] == construct]
                st = corr_stats(sub["d_construct"].to_numpy(), sub["dE"].to_numpy())
                mediation_rows.append({"arm": arm, "method": method, "run": run_name,
                                       "construct": construct, **st})

    spec_df = pd.DataFrame(specificity_rows)
    med_df = pd.DataFrame(mediation_rows)
    spec_df.to_csv(os.path.join(OUT_DIR, "specificity_net_deltas.csv"), index=False)
    med_df.to_csv(os.path.join(OUT_DIR, "mediation_correlations.csv"), index=False)

    print("=" * 78)
    print("SPECIFICITY -- net per-agent delta (t0 -> tN), saliency-blind vs canonical")
    print("=" * 78)
    print(spec_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print()
    print("=" * 78)
    print("MEDIATION -- pooled week-to-week corr(Δconstruct, ΔE[intention])")
    print("=" * 78)
    print(med_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print()
    print("How to read this:")
    print("- SPECIFICITY: compare mean_d_* across constructs within an arm/method. A policy")
    print("  hypothesised to move pbc (see sandbox/policy.py expected_pathways) should show")
    print("  mean_d_pbc as the largest mover in BOTH methods if the effect is content-driven,")
    print("  not just an artifact of relevance-filtered retrieval.")
    print("- MEDIATION: compare pearson_r per construct between saliency_blind and")
    print("  canonical_relevance for the SAME arm. If saliency_blind's correlations collapse")
    print("  toward 0 relative to canonical, the TPB<->intention link depended on retrieval")
    print("  pre-filtering by construct relevance -- the circularity risk this ablation")
    print("  exists to test (see docs/agent_memory/tpb-construct-validity-plan.md).")
    print(f"\nSaved -> {OUT_DIR}/specificity_net_deltas.csv, mediation_correlations.csv")


if __name__ == "__main__":
    main()
