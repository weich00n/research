"""Compare two C0 simulation runs (e.g. Qwen vs meta-Llama).

Answers two questions the runs were generated to settle:

  A. PROMPT ROBUSTNESS — given the *same* profile + prompt, do two different
     models produce the same *ideas*? Convergent validity of seed generation.
     Cross-model seed similarity is only meaningful against a control, so we
     contrast same-agent vs different-agent similarity (the gap = persona
     specificity), plus a uniform construct-composition agreement.

  B. MODEL QUALITY / BRAIN CHOICE — speed (from logs), JSON robustness, TPB
     construct coverage, and qualitative grounding — to pick the C0-C3 brain.

Read-only. Reuses utils.generate_utils.EmbeddingClient (all-MiniLM-L6-v2,
normalised -> cosine = dot) and sandbox.prompts.CONSTRUCT_PROMPTS. Mirrors the
conventions of compare_initialisations.py (load-by-agent_id, UTF-8 console,
CSV export).

Run from src/:
    python validation/compare_runs.py
    python validation/compare_runs.py \
        --runs ../outputs/runs/run_C0_Qwen.json ../outputs/runs/run_C0_metallama.json \
        --logs ../outputs/runs/run_C0_Qwen.log ../outputs/runs/run_C0_metallama.log \
        --labels qwen llama
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# this file lives in src/validation/; put src/ on the path so sandbox/utils
# import the same way they do for driver.py (which is run from src/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.prompts import CONSTRUCT_PROMPTS
from utils.generate_utils import EmbeddingClient

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "outputs")  # src/validation/ -> repo root
RUNS = os.path.join(OUT, "runs")
ANALYSIS = os.path.join(OUT, "analysis", "run_comparisons")
CONSTRUCTS = list(CONSTRUCT_PROMPTS)  # ['attitude', 'norm', 'pbc']
SAMPLE_AGENTS = ["agent_001", "agent_050", "agent_100"]  # for the grounding section
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"


# ── loading ────────────────────────────────────────────────────────────────

def load_run(path):
    """Return {agent_id: agent_dict} for one run JSON (engine.save output)."""
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    return {a["agent_id"]: a for a in state["agents"]}


def seed_texts(agent):
    """The agent's profile-seed memory texts (the generated background beliefs)."""
    return [m["memory_text"] for m in agent["memory_stream"]
            if m["source_type"] == "profile_seed"]


def expected_intention(dist):
    """E[intention] = sum_i (i+1) * p_i over the 5-level distribution (1..5)."""
    if not dist:
        return float("nan")
    return float(sum((i + 1) * p for i, p in enumerate(dist)))


# ── Part A: prompt robustness ───────────────────────────────────────────────

def align_sim(A, B):
    """Symmetric best-match alignment between two sets of (normalised) vectors.

    mean( for each row its best column , for each column its best row ) — a
    soft recall/precision of 'every idea in A has a close match in B and v.v.'.
    """
    S = A @ B.T
    return 0.5 * (S.max(axis=1).mean() + S.max(axis=0).mean())


def prompt_robustness(run_a, run_b, embed_cache, rng):
    """Same-agent vs different-agent cross-model seed similarity + construct mix.

    embed_cache maps text -> vector. Returns a per-agent DataFrame and a summary.
    """
    ids = sorted(run_a)
    cvecs = np.array([embed_cache[CONSTRUCT_PROMPTS[k]] for k in CONSTRUCTS])

    def vecs(run, aid):
        return np.array([embed_cache[t] for t in seed_texts(run[aid])])

    def construct_hist(V):
        # uniform construct tag per seed = argmax cosine to the construct prompts
        tags = (V @ cvecs.T).argmax(axis=1)
        return np.bincount(tags, minlength=3) / len(tags)

    rows = []
    for aid in ids:
        Va, Vb = vecs(run_a, aid), vecs(run_b, aid)
        same = align_sim(Va, Vb)
        others = [o for o in ids if o != aid]
        ctrl = np.mean([align_sim(Va, vecs(run_b, o))
                        for o in rng.choice(others, size=5, replace=False)])
        ha, hb = construct_hist(Va), construct_hist(Vb)
        tv = 0.5 * np.abs(ha - hb).sum()  # total-variation distance of construct mix
        rows.append({"agent_id": aid, "same_sim": same, "control_sim": ctrl,
                     "sim_gap": same - ctrl, "construct_tv": tv})
    df = pd.DataFrame(rows)

    same, ctrl = df["same_sim"], df["control_sim"]
    pooled_sd = np.sqrt((same.var(ddof=1) + ctrl.var(ddof=1)) / 2)
    summary = {
        "same_mean": same.mean(), "same_sd": same.std(ddof=1),
        "control_mean": ctrl.mean(), "control_sd": ctrl.std(ddof=1),
        "gap_mean": df["sim_gap"].mean(),
        "cohens_d": (same.mean() - ctrl.mean()) / pooled_sd if pooled_sd else float("nan"),
        "pct_same_gt_control": float((df["sim_gap"] > 0).mean() * 100),
        "construct_tv_mean": df["construct_tv"].mean(),
    }
    return df, summary


# ── Part B: model quality ───────────────────────────────────────────────────

def parse_log(path):
    """Timing + robustness from one run log, restricted to its LAST session.

    Logs are appended across runs; sessions are split on >300s gaps between
    consecutive timestamped lines, so the last session is the run we care about.
    """
    if not path or not os.path.exists(path):
        return None
    stamped = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if len(line) >= 19 and line[:4].isdigit():
            try:
                stamped.append((datetime.strptime(line[:19], LOG_TS_FMT), line))
            except ValueError:
                pass
    if not stamped:
        return None
    # split into sessions on >300s gaps, keep the last
    start = 0
    for i in range(1, len(stamped)):
        if (stamped[i][0] - stamped[i - 1][0]).total_seconds() > 300:
            start = i
    sess = stamped[start:]
    t0, tlast = sess[0][0], sess[-1][0]
    lines = [l for _, l in sess]
    text = "".join(lines)

    lat = [float(x) for x in re.findall(r"ok in ([\d.]+)s", text)]
    seed_ts = [t for t, l in sess if "seed memories" in l]
    step = re.search(r"Week \d+ done in (\d+)s", text)
    return {
        "total_s": (tlast - t0).total_seconds(),
        "seed_init_s": (seed_ts[-1] - t0).total_seconds() if seed_ts else None,
        "step_s": int(step.group(1)) if step else None,
        "lat_median": float(np.median(lat)) if lat else None,
        "lat_p90": float(np.percentile(lat, 90)) if lat else None,
        "n_calls": len(lat),
        "retried_calls": sum(1 for a in re.findall(r"attempt (\d+)", text) if int(a) > 1),
        "warn_parse": text.count("JSON parse failed"),
        "warn_empty": text.count("empty content"),
        "warn_failed": text.count("LLM call failed"),
        "warn_429": text.count("429 rate limited"),
    }


def json_validity(run):
    """Output-validity counts for one run (should be all-zero for a clean run)."""
    n_bad_dist = n_bad_tpb = n_short_seeds = 0
    for a in run.values():
        bs = a["belief_state"]
        d = bs.get("fertility_intention_dist")
        if not d or abs(sum(d) - 1) > 0.01:
            n_bad_dist += 1
        if not all(1 <= bs[k] <= 5 for k in
                   ("attitude_score", "subjective_norm_score", "pbc_score")):
            n_bad_tpb += 1
        if len(seed_texts(a)) < 5:
            n_short_seeds += 1
    return {"bad_dist": n_bad_dist, "bad_tpb": n_bad_tpb, "short_seeds": n_short_seeds}


def coverage_and_beliefs(run, embed_cache):
    """Construct coverage (uniform tags) + belief-state / E[intention] per agent."""
    cvecs = np.array([embed_cache[CONSTRUCT_PROMPTS[k]] for k in CONSTRUCTS])
    tag_counts = np.zeros(3)
    distinct = []
    beliefs = []
    for a in run.values():
        V = np.array([embed_cache[t] for t in seed_texts(a)])
        tags = (V @ cvecs.T).argmax(axis=1)
        tag_counts += np.bincount(tags, minlength=3)
        distinct.append(len(set(tags.tolist())))
        bs = a["belief_state"]
        beliefs.append({
            "agent_id": a["agent_id"],
            "attitude": bs["attitude_score"],
            "norm": bs["subjective_norm_score"],
            "pbc": bs["pbc_score"],
            "E_intention": expected_intention(bs.get("fertility_intention_dist")),
        })
    share = tag_counts / tag_counts.sum()
    return ({"share": {c: round(float(s), 3) for c, s in zip(CONSTRUCTS, share)},
             "mean_distinct_constructs": float(np.mean(distinct))},
            pd.DataFrame(beliefs))


# ── report ──────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs=2,
                    default=[os.path.join(RUNS, "run_C0.json"),
                             os.path.join(RUNS, "run_C0_metallama.json")])
    ap.add_argument("--logs", nargs=2,
                    default=[os.path.join(RUNS, "run_C0.log"),
                             os.path.join(RUNS, "run_C0_metallama.log")])
    ap.add_argument("--labels", nargs=2, default=["qwen", "llama"])
    ap.add_argument("--report", default=os.path.join(ANALYSIS, "run_comparison_qwen_vs_llama.md"))
    ap.add_argument("--csv", default=os.path.join(ANALYSIS, "run_comparison_per_agent.csv"))
    args = ap.parse_args()
    la, lb = args.labels

    run_a, run_b = load_run(args.runs[0]), load_run(args.runs[1])
    assert set(run_a) == set(run_b), "agent_id sets differ between runs"

    # embed every seed text (both runs) + the construct prompts, once
    all_texts = sorted({t for run in (run_a, run_b) for a in run.values()
                        for t in seed_texts(a)}
                       | set(CONSTRUCT_PROMPTS.values()))
    embed = EmbeddingClient()
    vecs = embed.embed(all_texts)
    cache = dict(zip(all_texts, vecs))

    rng = np.random.default_rng(42)
    robust_df, robust = prompt_robustness(run_a, run_b, cache, rng)
    logs = {la: parse_log(args.logs[0]), lb: parse_log(args.logs[1])}
    valid = {la: json_validity(run_a), lb: json_validity(run_b)}
    cov_a, bel_a = coverage_and_beliefs(run_a, cache)
    cov_b, bel_b = coverage_and_beliefs(run_b, cache)

    # paired downstream convergence
    bel = bel_a.merge(bel_b, on="agent_id", suffixes=(f"_{la}", f"_{lb}"))
    downstream = {}
    for col in ("attitude", "norm", "pbc", "E_intention"):
        x, y = bel[f"{col}_{la}"], bel[f"{col}_{lb}"]
        downstream[col] = {"mean_a": x.mean(), "mean_b": y.mean(),
                           "corr": float(np.corrcoef(x, y)[0, 1]),
                           "mean_abs_diff": float((x - y).abs().mean())}

    out = []
    p = out.append
    p(f"# C0 run comparison — {la} vs {lb}\n")
    p("Both runs: same 100 agents, same pipeline, tunneled identically "
      "(timing is a fair model-vs-model comparison).\n")

    p("## A. Prompt robustness (convergent validity of seed generation)\n")
    p(f"Cross-model seed similarity (cosine, all-MiniLM), paired by agent (n={len(robust_df)}):\n")
    p(f"- same-agent  {la}<->{lb}: **{robust['same_mean']:.3f}** (sd {robust['same_sd']:.3f})")
    p(f"- diff-agent (control)   : {robust['control_mean']:.3f} (sd {robust['control_sd']:.3f})")
    p(f"- **gap** (specificity)  : **{robust['gap_mean']:+.3f}**  | Cohen's d "
      f"{robust['cohens_d']:.2f} | same>control in {robust['pct_same_gt_control']:.0f}% of agents")
    p(f"- construct-mix divergence (TV, 0=identical): {robust['construct_tv_mean']:.3f}\n")
    verdict = ("seeds are persona-specific and consistent across models -> prompt drives output"
               if robust["cohens_d"] >= 0.8 and robust["gap_mean"] > 0
               else "WEAK separation — seeds may be generic, not persona-driven (inspect)")
    p(f"_Interpretation:_ {verdict}.\n")

    p("## B. Model quality / brain choice\n")
    p("### Speed (last log session)\n")
    p(f"| metric | {la} | {lb} |\n|---|---|---|")
    def g(d, k, suf=""):
        return "n/a" if not d or d.get(k) is None else f"{d[k]}{suf}"
    for key, lab, suf in [("seed_init_s", "seed-init", "s"), ("step_s", "t=1 step", "s"),
                          ("total_s", "total", "s"), ("lat_median", "median call", "s"),
                          ("lat_p90", "p90 call", "s"), ("n_calls", "calls", ""),
                          ("retried_calls", "retried calls", "")]:
        p(f"| {lab} | {g(logs[la], key, suf)} | {g(logs[lb], key, suf)} |")
    p("\n### Robustness (log warnings + JSON validity)\n")
    p(f"| metric | {la} | {lb} |\n|---|---|---|")
    for key, lab in [("warn_parse", "JSON parse-retries"), ("warn_empty", "empty-content retries"),
                     ("warn_failed", "call failures"), ("warn_429", "429s")]:
        p(f"| {lab} | {g(logs[la], key)} | {g(logs[lb], key)} |")
    for key, lab in [("bad_dist", "intention dist sum!=1"), ("bad_tpb", "TPB out of [1,5]"),
                     ("short_seeds", "agents with <5 seeds")]:
        p(f"| {lab} | {valid[la][key]} | {valid[lb][key]} |")
    p("\n### Construct coverage (uniform cosine tags)\n")
    p(f"- {la}: shares {cov_a['share']}, mean distinct constructs/agent "
      f"{cov_a['mean_distinct_constructs']:.2f}")
    p(f"- {lb}: shares {cov_b['share']}, mean distinct constructs/agent "
      f"{cov_b['mean_distinct_constructs']:.2f}\n")

    p("### Downstream convergence (paired belief state / E[intention])\n")
    p(f"| field | {la} mean | {lb} mean | corr | mean|diff| |\n|---|---|---|---|---|")
    for col, d in downstream.items():
        p(f"| {col} | {d['mean_a']:.2f} | {d['mean_b']:.2f} | {d['corr']:.2f} "
          f"| {d['mean_abs_diff']:.2f} |")

    p("\n### Grounding — sample seeds side by side\n")
    for aid in SAMPLE_AGENTS:
        if aid not in run_a:
            continue
        a = run_a[aid]
        p(f"**{aid}** — {a['age']}{a['gender'][0]}, {a['relationship_status']}, "
          f"{a['occupation']} ({a['planning_area']}), fin {a['financial_security_score']}\n")
        p(f"| # | {la} | {lb} |\n|---|---|---|")
        sa, sb = seed_texts(run_a[aid]), seed_texts(run_b[aid])
        for i in range(max(len(sa), len(sb))):
            ta = sa[i].replace("|", "/") if i < len(sa) else ""
            tb = sb[i].replace("|", "/") if i < len(sb) else ""
            p(f"| {i+1} | {ta} | {tb} |")
        p("")

    p("\n_Caveats: embedding similarity is topic not valence (hence the construct check); "
      "belief deltas reflect model + temperature-0.7 sampling + different seeds (descriptive). "
      "This is not the multi-LLM judge-IRR._")

    report = "\n".join(out)
    print(report)
    os.makedirs(OUT, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    robust_df.merge(bel, on="agent_id").to_csv(args.csv, index=False, encoding="utf-8")
    print(f"\nSaved report -> {args.report}\nSaved per-agent CSV -> {args.csv}")


if __name__ == "__main__":
    main()
