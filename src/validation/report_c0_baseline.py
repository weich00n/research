"""Saved report: the seed-initialised C0 baseline (seed -> t=1).

C0 is the frozen baseline — agents init at neutral 3/3/3 and take a *single*
belief update at t=1 off their seed memories only (no policy, no social). This
read-only script characterises that initialised state from a C0 run JSON: seed
counts, the t=1 TPB scores, and the fertility-intention distribution. It is the
reference point every later condition's deltas are measured against.

Defaults to the Qwen run (the chosen brain); `--run` points it at another model.

Reuses `compare_runs.py`'s loaders (`seed_texts`, `expected_intention`,
`json_validity`) so the numbers are computed exactly as elsewhere.

Run from src/:
    python validation/report_c0_baseline.py
    python validation/report_c0_baseline.py --run ../outputs/runs/run_C0_metallama.json --label llama
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import date

import numpy as np

# src/validation/ : add both src/ (for sandbox/utils that compare_runs needs)
# and this dir (for the compare_runs sibling import).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from compare_runs import expected_intention, json_validity, seed_texts  # noqa: E402

DEFAULT_RUN = os.path.join(ROOT, "outputs", "runs", "run_C0_Qwen.json")
DEFAULT_REPORT = os.path.join(ROOT, "reports", "c0_initialised_baseline.md")
DEFAULT_CSV = os.path.join(ROOT, "outputs", "analysis", "c0_baseline_per_agent.csv")

CONSTRUCTS = [("attitude", "attitude_score"),
              ("subjective norm", "subjective_norm_score"),
              ("PBC", "pbc_score")]
INTENTION_LEVELS = ["1 none", "2 weak", "3 uncertain", "4 likely", "5 strong"]


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max())}


def _grp_means(agents, key_fn):
    """{group: (n, mean_att, mean_norm, mean_pbc, mean_E)} over a grouping fn."""
    groups = {}
    buckets = {}
    for a in agents:
        buckets.setdefault(key_fn(a), []).append(a)
    for g, members in buckets.items():
        bs = [m["belief_state"] for m in members]
        E = [expected_intention(b.get("fertility_intention_dist")) for b in bs]
        groups[g] = (
            len(members),
            float(np.mean([b["attitude_score"] for b in bs])),
            float(np.mean([b["subjective_norm_score"] for b in bs])),
            float(np.mean([b["pbc_score"] for b in bs])),
            float(np.nanmean(E)),
        )
    return groups


def build_report(state, label, run_path, coverage=None):
    agents = state["agents"]
    run = {a["agent_id"]: a for a in agents}
    n = len(agents)
    out = []
    p = out.append

    p(f"# C0 Initialised Baseline — Seed → t=1 ({label})\n")
    p(f"**Date:** {date.today().isoformat()}  ")
    p(f"**Run:** `{os.path.relpath(run_path, ROOT)}`  ")
    p(f"**Condition:** {state.get('condition', 'C0')} "
      f"(static baseline, no policy/social) · "
      f"**timestep:** {state.get('current_timestep')} · **agents:** {n}\n")
    p("> C0 = the **frozen baseline**: agents initialise neutral (3/3/3) and take a "
      "*single* belief update at t=1 from their seed memories only. These numbers "
      "are the reference point that C1–C3 deltas are measured against.\n")

    # ── 1. Seed-memory init ─────────────────────────────────────────────────
    seed_counts = [len(seed_texts(a)) for a in agents]
    valid = json_validity(run)
    p("## 1. Seed-memory initialisation\n")
    p(f"- Total profile-seed memories: **{sum(seed_counts)}** "
      f"({np.mean(seed_counts):.1f}/agent; target 5).")
    p(f"- Agents with fewer than 5 seeds: **{valid['short_seeds']}**.\n")

    # ── 2. Baseline TPB at t=1 ──────────────────────────────────────────────
    p("## 2. Baseline TPB scores at t=1\n")
    p("Mean shift = movement off the neutral 3.0 init after one temperature-0.7 "
      "update.\n")
    p("| construct | mean | sd | min | max | shift vs 3.0 |")
    p("|---|---|---|---|---|---|")
    per_agent = {}
    for label_c, field in CONSTRUCTS:
        vals = [a["belief_state"][field] for a in agents]
        per_agent[field] = vals
        s = _stats(vals)
        shift = s["mean"] - 3.0
        if abs(shift) < 0.005:  # avoid an ugly "-0.00" from float noise
            shift = 0.0
        p(f"| {label_c} | {s['mean']:.2f} | {s['sd']:.2f} | {s['min']:.1f} | "
          f"{s['max']:.1f} | {shift:+.2f} |")
    # how much of the population actually moved
    moved_any = sum(
        1 for a in agents
        if any(abs(a["belief_state"][f] - 3.0) > 1e-9 for _, f in CONSTRUCTS))
    p(f"\n{moved_any}/{n} agents moved on ≥1 construct after t=1.\n")

    # ── 3. Fertility-intention distribution ─────────────────────────────────
    p("## 3. Fertility-intention distribution (t=1)\n")
    dists = [a["belief_state"].get("fertility_intention_dist") for a in agents]
    dists = [d for d in dists if d]
    mean_dist = np.mean(np.array(dists), axis=0)
    E = [expected_intention(d) for d in dists]
    p(f"Aggregate mean distribution over {len(dists)} agents "
      f"(sums to {mean_dist.sum():.3f}):\n")
    p("| level | mean p |")
    p("|---|---|")
    for lvl, pr in zip(INTENTION_LEVELS, mean_dist):
        p(f"| {lvl} | {pr:.3f} |")
    p(f"\n**E[intention] = {np.mean(E):.2f}** (sd {np.std(E, ddof=1):.2f}, "
      f"range {min(E):.2f}–{max(E):.2f}) on the 1–5 scale.\n")

    # ── 4. Context-validity cross-tabs ──────────────────────────────────────
    p("## 4. Context-validity cross-tabs\n")
    p("Descriptive only (a single t=1 step), but a plausibility check: do "
      "married agents and the financially-secure start higher?\n")

    def _table(title, groups, order):
        p(f"**By {title}**\n")
        p("| group | n | attitude | norm | PBC | E[intent] |")
        p("|---|---|---|---|---|---|")
        for g in order:
            if g in groups:
                ncnt, att, nrm, pbc, e = groups[g]
                p(f"| {g} | {ncnt} | {att:.2f} | {nrm:.2f} | {pbc:.2f} | {e:.2f} |")
        p("")

    by_marital = _grp_means(agents, lambda a: a.get("marital_status", "?"))
    _table("marital status", by_marital, ["Single", "Married"])
    by_fin = _grp_means(agents, lambda a: f"fin {a.get('financial_security_score', '?')}")
    _table("financial-security score", by_fin, ["fin 2", "fin 3", "fin 4"])

    if coverage is not None:
        p("**Seed construct coverage** (argmax cosine to construct prompts): "
          + ", ".join(f"{k} {v:.2f}" for k, v in coverage["share"].items())
          + f"; mean distinct constructs/agent {coverage['mean_distinct_constructs']:.2f}.\n")

    # ── 5. Output validity ──────────────────────────────────────────────────
    p("## 5. Output validity (expect all 0)\n")
    p("| check | count |")
    p("|---|---|")
    p(f"| intention dist sum ≠ 1 | {valid['bad_dist']} |")
    p(f"| TPB score out of [1,5] | {valid['bad_tpb']} |")
    p(f"| agents with < 5 seeds | {valid['short_seeds']} |")
    p("")

    # ── limitations + reproducibility ───────────────────────────────────────
    p("## Limitations\n")
    p(f"- A **single** temperature-0.7 update from a neutral start: belief spread "
      f"is small and **descriptive, not causal**. Cross-tab gaps are suggestive, "
      f"not tested.")
    p(f"- One model ({label}); see `reports/C0_model_comparison.md` for the "
      f"Qwen/Llama/Nemotron contrast.")
    p("- Demographic stratification of the pool is a separate report "
      "(`reports/agent_pool_mp_stratification.md`).\n")

    p("## Reproducibility / artifacts\n")
    p("- **Script:** `src/validation/report_c0_baseline.py` (read-only).")
    p("- **Per-agent CSV:** `outputs/analysis/c0_baseline_per_agent.csv`.")
    p("- **Loaders** (`seed_texts`, `expected_intention`, `json_validity`) "
      "imported from `src/validation/compare_runs.py`.")
    p(f"- **Regenerate:** from `src/`, "
      f"`python validation/report_c0_baseline.py --run {os.path.relpath(run_path, os.path.join(ROOT, 'src'))} --label {label}`")

    # per-agent CSV rows
    csv_rows = []
    for a in agents:
        bs = a["belief_state"]
        csv_rows.append([
            a["agent_id"], bs["attitude_score"], bs["subjective_norm_score"],
            bs["pbc_score"], expected_intention(bs.get("fertility_intention_dist")),
            len(seed_texts(a)),
        ])
    return "\n".join(out), csv_rows


def maybe_coverage(state):
    """Optional seed construct-coverage (needs sentence-transformers)."""
    from compare_runs import coverage_and_beliefs  # local: avoids import cost when off
    from sandbox.prompts import CONSTRUCT_PROMPTS
    from utils.generate_utils import EmbeddingClient

    run = {a["agent_id"]: a for a in state["agents"]}
    texts = sorted({t for a in run.values() for t in seed_texts(a)}
                   | set(CONSTRUCT_PROMPTS.values()))
    cache = dict(zip(texts, EmbeddingClient().embed(texts)))
    cov, _ = coverage_and_beliefs(run, cache)
    return cov


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--label", default="qwen")
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--with-construct-coverage", action="store_true",
                    help="add seed construct coverage (loads sentence-transformers)")
    args = ap.parse_args()

    with open(args.run, encoding="utf-8") as f:
        state = json.load(f)

    coverage = maybe_coverage(state) if args.with_construct_coverage else None
    report, csv_rows = build_report(state, args.label, args.run, coverage)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_id", "attitude", "norm", "pbc", "E_intention", "n_seeds"])
        w.writerows(csv_rows)

    print(report)
    print(f"\nSaved report -> {os.path.relpath(args.report, ROOT)}")
    print(f"Saved per-agent CSV -> {os.path.relpath(args.csv, ROOT)}")


if __name__ == "__main__":
    main()
