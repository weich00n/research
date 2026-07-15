"""News sanity eval: per-article TPB effects in isolation (VacSim eval mode 4).

One *cell* = one corpus article shown once (week 1) to a fresh copy of the
same N agents, with everything else off (no other news, no social posts),
run for --weeks weekly update cycles. A no-news control cell (condition C0,
same agents, same weeks) measures the no-input update drift; every article's
per-construct delta is reported raw and net of control.

Answers, before burning full runs: which article types over-drive attitude,
whether each policy moves its hypothesised construct (stimulus-level
specificity, cf. expected_pathways), and — once a context/"bad news" corpus
exists — whether negative articles push constructs down (unknown policy
names get a stub context Policy, so that corpus works here unchanged).

Usage (from src/, vLLM up; full 45-cell sweep ~1-1.5 h, resumable):
    python validation/news_sanity_eval.py
    python validation/news_sanity_eval.py --type family_impact --num-agents 5 --weeks 1
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):  # Windows cp1252 console can't print Δ
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engines.engine import Simulation
from LLM_judge import RelevanceScorer
from sandbox.agent import load_agents
from sandbox.lesson import reseed_id_counter as reseed_lesson_ids
from sandbox.news import News
from sandbox.policy import Policy, get_policies
from sandbox.tweet import reseed_id_counter as reseed_tweet_ids
from utils.generate_utils import LLMClient
from validation.compare_runs import expected_intention

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
DEFAULT_CORPUS = os.path.join(ROOT, "outputs", "news", "news_corpus_qwen.json")
DEFAULT_AGENTS = os.path.join(ROOT, "agents_final_100_seeded.json")
RUNS_DIR = os.path.join(ROOT, "outputs", "runs", "sanity")
ANALYSIS_DIR = os.path.join(ROOT, "outputs", "analysis", "news_validation")

CONSTRUCTS = [("attitude", "attitude_score"),
              ("norm", "subjective_norm_score"),
              ("pbc", "pbc_score")]
CONTROL = "sanity_control"


def load_cells(corpus_path, ids=None, policy=None, article_type=None):
    """Corpus articles filtered down to the requested cells."""
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"] if isinstance(data, dict) else data
    if ids:
        articles = [a for a in articles if a["news_id"] in set(ids)]
    if policy:
        articles = [a for a in articles if policy.lower() in a["policy_name"].lower()]
    if article_type:
        articles = [a for a in articles if a["article_type"] == article_type]
    return articles


def policy_for(name):
    """The registered Policy for an article, or a stub for context articles
    (e.g. future cost-of-living news) that aren't policy instruments."""
    for p in get_policies():
        if p.name == name:
            return p
    return Policy(name=name, category="context", description="",
                  expected_pathways=[])


def fresh_agents(path, n):
    """A pristine copy of the first n seeded agents, id counters reseeded
    (same pattern as driver.py) so freshly minted ids don't collide."""
    agents = load_agents(path)[:n]
    reseed_lesson_ids([l for a in agents for l in a.lessons])
    reseed_tweet_ids([t for a in agents for t in a.tweets])
    return agents


def cell_done(run_name, weeks):
    path = os.path.join(RUNS_DIR, f"{run_name}.json")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("current_timestep", 0) >= weeks


def run_cell(run_name, condition, news_schedule, args, llm, scorer):
    """Run one cell to completion (skipped if its run JSON is already done)."""
    if cell_done(run_name, args.weeks):
        print(f"  {run_name}: already complete, skipping")
        return
    agents = fresh_agents(args.agents, args.num_agents)
    sim = Simulation(agents=agents, network={}, condition=condition, llm=llm,
                     scorer=scorer, news_schedule=news_schedule,
                     output_dir=RUNS_DIR, run_name=run_name, verbose=False,
                     concurrency=args.concurrency)
    sim.initialise_baseline()  # no-op on pre-seeded agents
    for t in range(1, args.weeks + 1):
        sim.step(t)


def cell_metrics(run_name):
    """Mean/std per-construct delta (t=0 baseline -> final week) across agents."""
    with open(os.path.join(RUNS_DIR, f"{run_name}.json"), encoding="utf-8") as f:
        state = json.load(f)
    deltas = {label: [] for label, _ in CONSTRUCTS}
    deltas["intention"] = []
    for a in state["agents"]:
        hist = sorted(a["belief_history"], key=lambda h: h["timestep"])
        h0 = next(h for h in hist if h["timestep"] == 0)
        hN = hist[-1]
        for label, key in CONSTRUCTS:
            deltas[label].append(hN[key] - h0[key])
        deltas["intention"].append(
            expected_intention(hN["fertility_intention_dist"])
            - expected_intention(h0["fertility_intention_dist"]))
    return ({k: float(np.mean(v)) for k, v in deltas.items()},
            {k: float(np.std(v)) for k, v in deltas.items()},
            len(state["agents"]))


def build_report(articles, args):
    """Assemble rows (net of control), write md + csv, return the md text."""
    control_mean, control_std, n = cell_metrics(CONTROL)

    rows = []
    for a in articles:
        mean, std, _ = cell_metrics(f"sanity_{a['news_id']}")
        net = {k: mean[k] - control_mean[k] for k in mean}
        expected = policy_for(a["policy_name"]).expected_pathways
        top = max(("attitude", "norm", "pbc"), key=lambda c: abs(net[c]))
        rows.append({
            "news_id": a["news_id"], "policy": a["policy_name"],
            "type": a["article_type"], "mean": mean, "std": std, "net": net,
            "top": top, "expected": expected,
            "match": (top in expected) if expected else None,
        })
    rows.sort(key=lambda r: -max(abs(r["net"][c]) for c in ("attitude", "norm", "pbc")))

    lines = [
        "# News sanity eval (per-article TPB effects, isolation runs)\n",
        f"Agents per cell: {n} (first {n} of the shared seeded pool) · "
        f"weeks: {args.weeks} (exposure at week 1) · corpus: "
        f"{os.path.basename(args.corpus)} · deltas are t=0 -> week {args.weeks}, "
        f"net of the no-news control cell.\n",
        f"Control drift (raw): att {control_mean['attitude']:+.2f} "
        f"norm {control_mean['norm']:+.2f} pbc {control_mean['pbc']:+.2f} "
        f"E[int] {control_mean['intention']:+.2f}\n",
        "| article | type | net Δatt | net Δnorm | net Δpbc | net ΔE[int] "
        "| top | expected | match |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        exp = "/".join(r["expected"]) or "—"
        match = {True: "yes", False: "**NO**", None: "—"}[r["match"]]
        lines.append(
            f"| {r['news_id']} | {r['type']} | {r['net']['attitude']:+.2f} "
            f"| {r['net']['norm']:+.2f} | {r['net']['pbc']:+.2f} "
            f"| {r['net']['intention']:+.2f} | {r['top']} | {exp} | {match} |")

    for group_key, title in [("type", "article type"), ("policy", "policy")]:
        groups = {}
        for r in rows:
            groups.setdefault(r[group_key], []).append(r)
        lines += [f"\n## Mean net delta by {title}\n",
                  f"| {title} | n | Δatt | Δnorm | Δpbc | ΔE[int] |",
                  "|---|---|---|---|---|---|"]
        for g, rs in sorted(groups.items()):
            m = {c: float(np.mean([r["net"][c] for r in rs]))
                 for c in ("attitude", "norm", "pbc", "intention")}
            lines.append(f"| {g} | {len(rs)} | {m['attitude']:+.2f} "
                         f"| {m['norm']:+.2f} | {m['pbc']:+.2f} "
                         f"| {m['intention']:+.2f} |")

    md = "\n".join(lines) + "\n"
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    suffix = f"_{args.label}" if getattr(args, "label", "") else ""
    md_path = os.path.join(ANALYSIS_DIR, f"news_sanity_eval{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    csv_path = os.path.join(ANALYSIS_DIR, f"news_sanity_eval{suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["news_id", "policy", "article_type",
                    "raw_att", "raw_norm", "raw_pbc", "raw_int",
                    "net_att", "net_norm", "net_pbc", "net_int",
                    "std_att", "std_norm", "std_pbc",
                    "top_construct", "expected", "match"])
        for r in rows:
            w.writerow([r["news_id"], r["policy"], r["type"],
                        *(round(r["mean"][c], 4) for c in ("attitude", "norm", "pbc", "intention")),
                        *(round(r["net"][c], 4) for c in ("attitude", "norm", "pbc", "intention")),
                        *(round(r["std"][c], 4) for c in ("attitude", "norm", "pbc")),
                        r["top"], "/".join(r["expected"]), r["match"]])
    return md, md_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--agents", default=DEFAULT_AGENTS)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--weeks", type=int, default=2,
                        help="update cycles per cell (exposure at week 1 only)")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--relevance-mode", choices=["llm", "cosine", "hybrid"],
                        default="llm")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="only these news_ids")
    parser.add_argument("--policy", default=None,
                        help="only articles whose policy name contains this")
    parser.add_argument("--type", dest="article_type", default=None,
                        help="only this article_type")
    parser.add_argument("--label", default="",
                        help="suffix for the report filenames (e.g. 'legacy' -> "
                             "news_sanity_eval_legacy.md) so different corpora "
                             "don't overwrite each other's reports")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild the report from existing cell runs, no LLM")
    args = parser.parse_args()

    articles = load_cells(args.corpus, args.ids, args.policy, args.article_type)
    if not articles:
        raise SystemExit("no articles match the given filters")
    print(f"{len(articles)} article cells + 1 control "
          f"({args.num_agents} agents x {args.weeks} weeks each)")

    if not args.report_only:
        llm = LLMClient()
        print(f"LLM: {llm.provider} / {llm.model}")
        scorer = RelevanceScorer(mode=args.relevance_mode, llm=llm)
        os.makedirs(RUNS_DIR, exist_ok=True)

        run_cell(CONTROL, "C0", None, args, llm, scorer)
        for i, a in enumerate(articles, 1):
            print(f"[{i}/{len(articles)}] {a['news_id']}")
            news = News(policy_for(a["policy_name"]), 1, text=a["text"],
                        news_id=a["news_id"], article_type=a["article_type"])
            run_cell(f"sanity_{a['news_id']}", "C2", {1: [news]}, args, llm, scorer)

    if not cell_done(CONTROL, args.weeks):
        raise SystemExit(f"control cell missing/incomplete under {RUNS_DIR} - "
                         f"run without --report-only first")
    missing = [a["news_id"] for a in articles
               if not cell_done(f"sanity_{a['news_id']}", args.weeks)]
    if missing:
        raise SystemExit(f"{len(missing)} cells missing/incomplete "
                         f"(e.g. {missing[:3]}) - run without --report-only first")

    md, md_path, csv_path = build_report(articles, args)
    print("\n" + md)
    print(f"saved -> {md_path}\nsaved -> {csv_path}")


if __name__ == "__main__":
    main()
