"""Three-way comparison: TPB mediation study vs VacSim-direct (no-TPB, fixed
schedule) vs VacSim-recsys (no-TPB, recommender-selected exposure).

Matches all 10 condition-arms across all three methodologies (the direct
and recsys arms are named identically across those two; the TPB study's
matching file was picked by hand -- see MATCHED_ARMS below -- since its ~30
run files span multiple iterations with no policy-category/corpus metadata
recorded in the JSON itself).

Produces, under outputs/analysis/run_comparisons/three_way/:
  - combined_trajectories.csv   methodology x arm x week -> mean E[intention]
  - arm_summary.csv             per methodology x arm: deltas, saturation, validity, log robustness
  - paired_stats.csv            per arm, paired (same-agent) comparison between methodology pairs, on t0/tN/delta

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
SEEDED_AGENTS_PATH = os.path.join(REPO, "agents_final_100_seeded.json")

CEILING, FLOOR = 4.8, 1.2
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"

# (arm label, TPB run name or None, direct run name, recsys run name,
#  TPB run dir override or None -> defaults to RUNS)
MATCHED_ARMS = [
    # run_C0_Qwen.json predates agents_final_100_seeded.json's freeze (mtime
    # Jun 25 vs Jul 3) and its t0 mismatches the frozen baseline on 99/100
    # agents -- it's a stale run from an earlier pipeline, NOT a valid C0.
    # c0_smoke_gated.json (Jul 14, in smoke/) matches the frozen baseline
    # exactly on all 100 agents and stays flat for all 12 weeks as a true
    # static condition should -- verified bit-for-bit, use this instead.
    ("C0_static",            "c0_smoke_gated",    "run_C0_direct",                    "run_C0_recsys",                    SMOKE),
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
    ("C3_context_headwind",  "run_C3_headwind",   "run_C3_direct_context_headwind",   "run_C3_recsys_context_headwind",   None),
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


def delta_trajectory(state, methodology):
    """Per-week mean(E_w - E_t0) per agent -- the trajectory re-centered at 0
    for every agent at their own t0. This is what makes the SHAPE of the
    three methodologies' trajectories comparable despite the baseline
    mismatch: every line starts at (week=0, delta=0) by construction, so any
    visible divergence is real weekly dynamics, not a baseline offset."""
    hist_fn = tpb_history if methodology == "tpb" else direct_history
    by_week = {}
    for a in state["agents"]:
        hist = {h["timestep"]: h["E"] for h in hist_fn(a)}
        if 0 not in hist:
            continue
        base = hist[0]
        for w, e in hist.items():
            by_week.setdefault(w, []).append(e - base)
    return {w: float(np.mean(vals)) for w, vals in sorted(by_week.items())}


def final_agent_values(state, methodology, final_t):
    hist_fn = tpb_history if methodology == "tpb" else direct_history
    out = {}
    for a in state["agents"]:
        hist = {h["timestep"]: h["E"] for h in hist_fn(a)}
        if final_t in hist:
            out[a["agent_id"]] = hist[final_t]
    return out


def agent_values_at(state, methodology, t):
    hist_fn = tpb_history if methodology == "tpb" else direct_history
    out = {}
    for a in state["agents"]:
        hist = {h["timestep"]: h["E"] for h in hist_fn(a)}
        if t in hist:
            out[a["agent_id"]] = hist[t]
    return out


def agent_deltas(state, methodology, t0, tN):
    """Per-agent (E_tN - E_t0), the within-methodology change -- the only
    quantity NOT confounded by the baseline-elicitation mismatch below."""
    v0 = agent_values_at(state, methodology, t0)
    vN = agent_values_at(state, methodology, tN)
    return {aid: vN[aid] - v0[aid] for aid in v0 if aid in vN}


def paired_compare(xa, xb):
    """Paired stats for two same-length, same-order arrays: mean diff, paired
    t-test, Wilcoxon, paired Cohen's d, and Pearson/Spearman correlation (do
    the two series even rank-agree, independent of their mean difference?)."""
    diff = xa - xb
    t_stat, t_p = stats.ttest_rel(xa, xb)
    try:
        w_stat, w_p = stats.wilcoxon(xa, xb)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
    if np.std(xa) > 0 and np.std(xb) > 0:
        pearson_r = float(np.corrcoef(xa, xb)[0, 1])
        spearman_r = float(stats.spearmanr(xa, xb).correlation)
    else:
        pearson_r = spearman_r = float("nan")
    return {
        "n_agents": len(xa), "mean_a": xa.mean(), "mean_b": xb.mean(), "mean_diff": diff.mean(),
        "t_stat": t_stat, "t_p": t_p, "wilcoxon_p": w_p, "cohens_d_paired": cohens_d,
        "pearson_r": pearson_r, "spearman_r": spearman_r,
    }


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


def load_seeded_baseline():
    """{agent_id: fertility_intention_dist} from the frozen TPB baseline file
    -- the ground truth every TPB comparator run's t0 MUST match exactly, or
    it isn't reading the same shared baseline CLAUDE.md says it does (see the
    run_C0_Qwen bug this check exists to catch: 99/100 agents mismatched
    because that run predated the baseline freeze by over a week)."""
    with open(SEEDED_AGENTS_PATH, encoding="utf-8") as f:
        seeded = json.load(f)
    return {a["agent_id"]: a["belief_state"]["fertility_intention_dist"] for a in seeded}


def validate_tpb_baseline(state, name, seeded_baseline):
    """Assert every agent's t0 belief_history entry matches the frozen
    baseline file bit-for-bit. Raises loudly rather than silently comparing
    two runs that were never actually on the same starting state."""
    mismatches = []
    for a in state["agents"]:
        aid = a["agent_id"]
        if aid not in seeded_baseline:
            continue
        hist = a.get("belief_history") or []
        if hist and hist[0].get("timestep") != 0:
            raise ValueError(
                f"{name}: agent {aid}'s belief_history[0] is timestep "
                f"{hist[0].get('timestep')}, not 0 -- history may be truncated "
                f"(e.g. from a resumed run); refusing to compare it as t0.")
        t0_dist = hist[0].get("fertility_intention_dist") if hist else None
        if t0_dist != seeded_baseline[aid]:
            mismatches.append(aid)
    if mismatches:
        raise ValueError(
            f"{name}: {len(mismatches)}/{len(state['agents'])} agents' t0 belief_history "
            f"does NOT match the frozen agents_final_100_seeded.json baseline (e.g. "
            f"{mismatches[:3]}...) -- this run is not comparable as a 'shared baseline' "
            f"TPB arm. Do not add it to MATCHED_ARMS without investigating why.")


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
        "NOTE: E_t0 is NOT a shared control across methodologies. TPB's E_t0\n"
        "comes from the frozen agents_final_100_seeded.json (belief_state\n"
        "schema); direct/recsys agents read a different top-level field that\n"
        "doesn't exist in that file and silently re-elicit their OWN t0 via a\n"
        "different prompt (engine_direct.py's _run_agent_baseline). Every TPB\n"
        "comparator run is now validated (validate_tpb_baseline, below) to\n"
        "confirm it actually matches the frozen file bit-for-bit -- this is\n"
        "what caught run_C0_Qwen.json being a stale pre-freeze run (99/100\n"
        "agent mismatch), now replaced with c0_smoke_gated.json (0 mismatches).\n"
        "Compare Δ (E_tN - E_t0, within methodology), not raw E_t0/E_tN.\n"
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    seeded_baseline = load_seeded_baseline()
    traj_rows, summary_rows, pair_rows = [], [], []
    final_values = {}   # (arm, methodology) -> {agent_id: E at tN}
    t0_values = {}       # (arm, methodology) -> {agent_id: E at t0}
    delta_values = {}    # (arm, methodology) -> {agent_id: E_tN - E_t0}
    final_tN = {}       # (arm, methodology) -> tN, for the pairing mismatch check
    n_tpb_validated = 0
    decomp_checks = [0, 0]  # [total, failed]

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
            if methodology == "tpb":
                validate_tpb_baseline(state, name, seeded_baseline)
                n_tpb_validated += 1
            traj = trajectory(state, methodology)
            dtraj = delta_trajectory(state, methodology)
            weeks = sorted(traj)
            t0, tN = weeks[0], weeks[-1]
            for w, e in traj.items():
                traj_rows.append({"arm": arm, "methodology": methodology, "week": w,
                                  "E_intention": e, "E_delta_from_own_t0": dtraj.get(w)})
            fvals = final_agent_values(state, methodology, tN)
            final_values[(arm, methodology)] = fvals
            t0_values[(arm, methodology)] = agent_values_at(state, methodology, t0)
            delta_values[(arm, methodology)] = agent_deltas(state, methodology, t0, tN)
            final_tN[(arm, methodology)] = tN
            vals_arr = np.array(list(fvals.values()))
            bad_dist, bad_tpb = dist_validity(state, methodology)
            log_path = os.path.join(dirs[methodology], f"{name}.log")
            log = parse_log(log_path) or {}
            summary_rows.append({
                "arm": arm, "methodology": methodology, "run": name,
                "n_agents": len(state["agents"]), "weeks_done": tN,
                "E_t0": traj[t0], "E_tN": traj[tN], "net_delta": traj[tN] - traj[t0],
                "headroom_t0": 5 - traj[t0], "legroom_t0": traj[t0] - 1,
                "sd_tN": float(np.std(vals_arr, ddof=1)) if len(vals_arr) > 1 else float("nan"),
                "ceiling_share_tN": float(np.mean(vals_arr >= CEILING)),
                "floor_share_tN": float(np.mean(vals_arr <= FLOOR)),
                "gated_share": gated_share(state, methodology),
                "bad_dist": bad_dist, "bad_tpb_range": bad_tpb,
                "log_n_calls": log.get("n_calls"), "log_lat_median_s": log.get("lat_median"),
                "log_warn_parse": log.get("warn_parse"), "log_warn_empty": log.get("warn_empty"),
                "log_warn_failed": log.get("warn_failed"), "log_warn_429": log.get("warn_429"),
            })

        # Paired stats: every pair of methodologies present for this arm,
        # matched by agent_id, on THREE metrics -- t0 (baseline-equivalence
        # check), tN (raw endpoint -- confounded, kept for reference), and
        # delta (E_tN - E_t0 per agent -- the within-methodology change,
        # unconfounded by baseline-elicitation differences). By construction
        # diff_tN == diff_t0 + diff_delta; asserted below as a consistency
        # check on the numbers themselves.
        present = [m for m in METHODOLOGIES if (arm, m) in final_values]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                ma, mb = present[i], present[j]
                tN_a, tN_b = final_tN[(arm, ma)], final_tN[(arm, mb)]
                tN_mismatch = tN_a != tN_b
                if tN_mismatch:
                    print(f"WARNING: {arm} {ma} (tN={tN_a}) vs {mb} (tN={tN_b}) "
                          f"-- final timesteps differ, pairing anyway but flagged")

                metric_sources = {"t0": t0_values, "tN": final_values, "delta": delta_values}
                metric_stats = {}
                for metric, source in metric_sources.items():
                    da, db = source[(arm, ma)], source[(arm, mb)]
                    shared = sorted(set(da) & set(db))
                    if len(shared) < 3:
                        continue
                    xa = np.array([da[k] for k in shared])
                    xb = np.array([db[k] for k in shared])
                    st = paired_compare(xa, xb)
                    metric_stats[metric] = st
                    pair_rows.append({
                        "arm": arm, "pair": f"{ma}_vs_{mb}", "metric": metric,
                        "tN_a": tN_a, "tN_b": tN_b, "tN_mismatch": tN_mismatch, **st,
                    })

                if {"t0", "tN", "delta"} <= metric_stats.keys():
                    lhs = metric_stats["tN"]["mean_diff"]
                    rhs = metric_stats["t0"]["mean_diff"] + metric_stats["delta"]["mean_diff"]
                    decomp_checks[0] += 1
                    if abs(lhs - rhs) > 1e-6:
                        decomp_checks[1] += 1
                        print(f"WARNING: {arm} {ma}_vs_{mb} decomposition identity "
                              f"failed: tN diff {lhs:.4f} != t0 diff + delta diff {rhs:.4f} "
                              f"(shared-agent sets differ between metrics -- check for missing weeks)")

    traj_df = pd.DataFrame(traj_rows)
    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)

    traj_df.to_csv(os.path.join(OUT_DIR, "combined_trajectories.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "arm_summary.csv"), index=False)
    pair_df.to_csv(os.path.join(OUT_DIR, "paired_stats.csv"), index=False)

    print(f"\n{n_tpb_validated}/{n_tpb_validated} TPB comparator runs validated against the "
          f"frozen baseline (bit-for-bit, 100 agents each) -- no silent staleness.")
    print(f"{decomp_checks[0] - decomp_checks[1]}/{decomp_checks[0]} decomposition identity "
          f"checks passed (raw endpoint diff == baseline diff + delta diff, tol 1e-6).")
    print(f"Saved {len(traj_df)} trajectory rows, {len(summary_df)} summary rows, "
          f"{len(pair_df)} paired-stat rows -> {OUT_DIR}")
    return traj_df, summary_df, pair_df


if __name__ == "__main__":
    main()
