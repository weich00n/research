"""Three-way comparison: TPB mediation study vs VacSim-direct (no-TPB, fixed
schedule) vs VacSim-recsys (no-TPB, recommender-selected exposure).

Matches 9 of the 10 condition-arms across all three methodologies (the direct
and recsys arms are named identically across those two; the TPB study's
matching file was picked by hand -- see MATCHED_ARMS below -- since its ~30
run files span multiple iterations with no policy-category/corpus metadata
recorded in the JSON itself). C3_context_headwind has no TPB counterpart (was
never run under that methodology) and is reported as missing, not estimated.

Produces, under outputs/analysis/run_comparisons/three_way/:
  - combined_trajectories.csv   methodology x arm x week -> mean E[intention]
  - arm_summary.csv             per methodology x arm: deltas, saturation, validity, log robustness
  - paired_stats.csv            per arm, paired (same-agent) comparison between methodology pairs at the final week
  - three_way_comparison.md     the human-readable report tying it together

Run from src/:
    python validation/compare_three_methodologies.py
"""

import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
RUNS = os.path.join(REPO, "outputs", "runs")
SMOKE = os.path.join(RUNS, "smoke")
OUT_DIR = os.path.join(REPO, "outputs", "analysis", "run_comparisons", "three_way")

CEILING, FLOOR = 4.8, 1.2
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"

# (arm label, TPB run name or None, direct run name, recsys run name,
#  TPB run dir override or None -> defaults to RUNS)
MATCHED_ARMS = [
    ("C0_static",            "run_C0_Qwen",       "run_C0_direct",                    "run_C0_recsys",                    None),
    ("C1_social_only",       "run_C1_Qwen_fixed", "run_C1_direct",                    "run_C1_recsys",                    None),
    ("C2_news_only",         "c2_smoke_corpus",   "run_C2_direct_news_only",          "run_C2_recsys_news_only",          SMOKE),
    ("C2_ambient_combined",  "run_C2_ambient",    "run_C2_direct_ambient_combined",   "run_C2_recsys_ambient_combined",   None),
    ("C2_context_only",      "run_context_only",  "run_C2_direct_context_only",       "run_C2_recsys_context_only",       None),
    ("C2_context_headwind",  "run_C2_headwind",   "run_C2_direct_context_headwind",   "run_C2_recsys_context_headwind",   None),
    ("C3_news_only",         "run_C3_Qwen",       "run_C3_direct_news_only",          "run_C3_recsys_news_only",          None),
    ("C3_ambient_combined",  "run_C3_ambient",    "run_C3_direct_ambient_combined",   "run_C3_recsys_ambient_combined",   None),
    # run_C1_ambient.json is internally condition=C3 (needs policy_on=True to
    # read the schedule) but --context-corpus with no --news-corpus makes the
    # schedule context-only -- i.e. social+ambient, no real policy. That is
    # exactly what the direct/recsys "C3_context_only" arms are.
    ("C3_context_only",      "run_C1_ambient",    "run_C3_direct_context_only",       "run_C3_recsys_context_only",       None),
    ("C3_context_headwind",  None,                "run_C3_direct_context_headwind",   "run_C3_recsys_context_headwind",   None),
]

METHODOLOGIES = ["tpb", "direct", "recsys"]


def expected_intention(dist):
    if not dist:
        return float("nan")
    return float(sum((i + 1) * p for i, p in enumerate(dist)))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── per-methodology history readers ─────────────────────────────────────────

def tpb_history(agent):
    return [{"timestep": h["timestep"], "E": expected_intention(h.get("fertility_intention_dist"))}
            for h in agent.get("belief_history", [])]


def direct_history(agent):
    return [{"timestep": h["timestep"], "E": expected_intention(h.get("fertility_intention_dist"))}
            for h in agent.get("intention_history", [])]


def trajectory(state, methodology):
    hist_fn = tpb_history if methodology == "tpb" else direct_history
    by_week = {}
    for a in state["agents"]:
        for h in hist_fn(a):
            by_week.setdefault(h["timestep"], []).append(h["E"])
    return {w: float(np.mean(vals)) for w, vals in sorted(by_week.items())}


def final_agent_values(state, methodology, final_t):
    hist_fn = tpb_history if methodology == "tpb" else direct_history
    out = {}
    for a in state["agents"]:
        hist = {h["timestep"]: h["E"] for h in hist_fn(a)}
        if final_t in hist:
            out[a["agent_id"]] = hist[final_t]
    return out


def dist_validity(state, methodology):
    """Count agents whose LATEST intention distribution doesn't sum to 1
    (~0.01 tol) or (TPB only) whose TPB scores fall outside [1,5]."""
    bad_dist = bad_tpb = 0
    for a in state["agents"]:
        if methodology == "tpb":
            bs = a["belief_state"]
            d = bs.get("fertility_intention_dist")
            if not all(1 <= bs[k] <= 5 for k in
                       ("attitude_score", "subjective_norm_score", "pbc_score")):
                bad_tpb += 1
        else:
            d = a.get("fertility_intention_dist")
        if not d or abs(sum(d) - 1) > 0.01:
            bad_dist += 1
    return bad_dist, bad_tpb


def gated_share(state, methodology):
    hist_fn = tpb_history if methodology == "tpb" else direct_history
    if methodology == "tpb":
        return float("nan")  # TPB engine has no equivalent gate/reasoning field on belief_history
    total = gated = 0
    for a in state["agents"]:
        for h in a.get("intention_history", []):
            if h["timestep"] == 0:
                continue
            total += 1
            if h.get("reasoning", "").startswith("(gated"):
                gated += 1
    return gated / total if total else float("nan")


def parse_log(path):
    """Timing + robustness from one run log, restricted to its LAST session
    (logs are appended across resumed runs; split on >300s gaps)."""
    if not path or not os.path.exists(path):
        return None
    stamped = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if len(line) >= 19 and line[:4].isdigit():
            try:
                from datetime import datetime
                stamped.append((datetime.strptime(line[:19], LOG_TS_FMT), line))
            except ValueError:
                pass
    if not stamped:
        return None
    start = 0
    for i in range(1, len(stamped)):
        if (stamped[i][0] - stamped[i - 1][0]).total_seconds() > 300:
            start = i
    sess = stamped[start:]
    text = "".join(l for _, l in sess)
    lat = [float(x) for x in re.findall(r"ok in ([\d.]+)s", text)]
    return {
        "n_calls": len(lat),
        "lat_median": float(np.median(lat)) if lat else None,
        "warn_parse": text.count("JSON parse failed"),
        "warn_empty": text.count("empty content"),
        "warn_failed": text.count("LLM call failed"),
        "warn_429": text.count("429 rate limited"),
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print(
        "NOTE: E_t0/E_tN are each methodology's OWN baseline, not a shared\n"
        "control. TPB's E_t0 comes from the frozen, shared\n"
        "agents_final_100_seeded.json; direct/recsys agents build their own\n"
        "fresh t=0 baseline via a different prompt (engine_direct.py's\n"
        "_run_agent_baseline) -- this is intentional per-methodology design,\n"
        "not a shared control condition. Comparing raw E_t0/E_tN across\n"
        "methodologies risks attributing baseline-prompt wording differences\n"
        "to methodology effects. net_delta (E_tN - E_t0, computed within each\n"
        "methodology) is the only quantity that's actually comparable\n"
        "across methodologies.\n"
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    traj_rows, summary_rows, pair_rows = [], [], []
    final_values = {}  # (arm, methodology) -> {agent_id: E}
    final_tN = {}       # (arm, methodology) -> tN, for the pairing mismatch check

    for arm, tpb_name, direct_name, recsys_name, tpb_dir in MATCHED_ARMS:
        names = {"tpb": tpb_name, "direct": direct_name, "recsys": recsys_name}
        dirs = {"tpb": tpb_dir or RUNS, "direct": RUNS, "recsys": RUNS}
        for methodology in METHODOLOGIES:
            name = names[methodology]
            if not name:
                print(f"  {arm} / {methodology}: no counterpart -- skipping")
                continue
            path = os.path.join(dirs[methodology], f"{name}.json")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}, skipping {arm}/{methodology}")
                continue
            state = load_json(path)
            traj = trajectory(state, methodology)
            weeks = sorted(traj)
            t0, tN = weeks[0], weeks[-1]
            for w, e in traj.items():
                traj_rows.append({"arm": arm, "methodology": methodology, "week": w, "E_intention": e})
            fvals = final_agent_values(state, methodology, tN)
            final_values[(arm, methodology)] = fvals
            final_tN[(arm, methodology)] = tN
            vals_arr = np.array(list(fvals.values()))
            bad_dist, bad_tpb = dist_validity(state, methodology)
            log_path = os.path.join(dirs[methodology], f"{name}.log")
            log = parse_log(log_path) or {}
            summary_rows.append({
                "arm": arm, "methodology": methodology, "run": name,
                "n_agents": len(state["agents"]), "weeks_done": tN,
                "E_t0": traj[t0], "E_tN": traj[tN], "net_delta": traj[tN] - traj[t0],
                "sd_tN": float(np.std(vals_arr, ddof=1)) if len(vals_arr) > 1 else float("nan"),
                "ceiling_share_tN": float(np.mean(vals_arr >= CEILING)),
                "floor_share_tN": float(np.mean(vals_arr <= FLOOR)),
                "gated_share": gated_share(state, methodology),
                "bad_dist": bad_dist, "bad_tpb_range": bad_tpb,
                "log_n_calls": log.get("n_calls"), "log_lat_median_s": log.get("lat_median"),
                "log_warn_parse": log.get("warn_parse"), "log_warn_empty": log.get("warn_empty"),
                "log_warn_failed": log.get("warn_failed"), "log_warn_429": log.get("warn_429"),
            })

        # paired stats: every pair of methodologies present for this arm, matched by agent_id
        present = [m for m in METHODOLOGIES if (arm, m) in final_values]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                ma, mb = present[i], present[j]
                fa, fb = final_values[(arm, ma)], final_values[(arm, mb)]
                shared = sorted(set(fa) & set(fb))
                if len(shared) < 3:
                    continue
                xa = np.array([fa[k] for k in shared])
                xb = np.array([fb[k] for k in shared])
                diff = xa - xb
                t_stat, t_p = stats.ttest_rel(xa, xb)
                try:
                    w_stat, w_p = stats.wilcoxon(xa, xb)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
                cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
                # Both sides' "final week" should be the same absolute
                # timestep, or this pairing is comparing different depths
                # (e.g. one side resumed less far than the other) rather than
                # each run's true endpoint. Flag rather than silently pair.
                tN_a, tN_b = final_tN[(arm, ma)], final_tN[(arm, mb)]
                tN_mismatch = tN_a != tN_b
                if tN_mismatch:
                    print(f"WARNING: {arm} {ma} (tN={tN_a}) vs {mb} (tN={tN_b}) "
                          f"-- final timesteps differ, pairing anyway but flagged")
                pair_rows.append({
                    "arm": arm, "pair": f"{ma}_vs_{mb}", "n_agents": len(shared),
                    "mean_a": xa.mean(), "mean_b": xb.mean(), "mean_diff": diff.mean(),
                    "t_stat": t_stat, "t_p": t_p, "wilcoxon_p": w_p, "cohens_d_paired": cohens_d,
                    "tN_a": tN_a, "tN_b": tN_b, "tN_mismatch": tN_mismatch,
                })

    traj_df = pd.DataFrame(traj_rows)
    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)

    traj_df.to_csv(os.path.join(OUT_DIR, "combined_trajectories.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "arm_summary.csv"), index=False)
    pair_df.to_csv(os.path.join(OUT_DIR, "paired_stats.csv"), index=False)

    print(f"Saved {len(traj_df)} trajectory rows, {len(summary_df)} summary rows, "
          f"{len(pair_df)} paired-stat rows -> {OUT_DIR}")
    return traj_df, summary_df, pair_df


if __name__ == "__main__":
    main()
