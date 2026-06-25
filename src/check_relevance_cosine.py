"""Validate stored LLM-judge relevance scores against an independent cosine signal.

Every memory carries a `relevance` dict — the LLM-as-judge attitude/norm/pbc
scores (RelevanceScorer mode='llm'). Here we recompute relevance the *other*
way — cosine similarity of the memory text to the TPB construct prompts (the
same computation RelevanceScorer mode='cosine' uses) — and measure convergence:

  - per-construct Spearman + Pearson correlation (LLM vs cosine)
  - argmax (dominant construct) agreement: accuracy + Cohen's kappa

High agreement = convergent validity from a methodologically different signal.
Disagreement is AMBIGUOUS: cosine-to-construct-prompts is itself weak, so it is
a triangulation check, not an arbiter. The multi-LLM judge-IRR remains primary.

Read-only. Run from src/:
    python check_relevance_cosine.py
    python check_relevance_cosine.py \
        --runs ../outputs/run_C0.json ../outputs/run_C0_metallama.json --labels qwen llama
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from sandbox.prompts import CONSTRUCT_PROMPTS
from utils.generate_utils import EmbeddingClient

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
CONSTRUCTS = list(CONSTRUCT_PROMPTS)  # ['attitude', 'norm', 'pbc']


def collect_memories(path):
    """All memories carrying a relevance dict, as rows (one per memory)."""
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    rows = []
    for a in state["agents"]:
        for m in a["memory_stream"]:
            rel = m.get("relevance")
            if not rel:
                continue
            rows.append({"agent_id": a["agent_id"], "memory_id": m["memory_id"],
                         "source_type": m["source_type"], "text": m["memory_text"],
                         **{f"llm_{k}": float(rel.get(k, 0.0)) for k in CONSTRUCTS}})
    return pd.DataFrame(rows)


def cosine_relevance(texts, embed, cvecs):
    """Cosine relevance to each construct prompt (mirrors RelevanceScorer cosine mode)."""
    vecs = embed.embed(list(texts))            # L2-normalised -> dot = cosine
    return np.clip(vecs @ cvecs.T, 0.0, 1.0)   # (N, 3)


def evaluate(df):
    """Per-construct correlations + argmax agreement between llm_* and cos_* columns."""
    per_construct = {}
    for k in CONSTRUCTS:
        x, y = df[f"llm_{k}"], df[f"cos_{k}"]
        per_construct[k] = {
            "pearson": float(x.corr(y, method="pearson")),
            "spearman": float(x.corr(y, method="spearman")),
            "mean_abs_diff": float((x - y).abs().mean()),
        }
    llm_arg = df[[f"llm_{k}" for k in CONSTRUCTS]].values.argmax(axis=1)
    cos_arg = df[[f"cos_{k}" for k in CONSTRUCTS]].values.argmax(axis=1)
    argmax = {
        "accuracy": float((llm_arg == cos_arg).mean()),
        "kappa": float(cohen_kappa_score(llm_arg, cos_arg)),
        "mean_spearman": float(np.mean([per_construct[k]["spearman"] for k in CONSTRUCTS])),
    }
    df = df.assign(llm_argmax=[CONSTRUCTS[i] for i in llm_arg],
                   cos_argmax=[CONSTRUCTS[i] for i in cos_arg],
                   argmax_match=(llm_arg == cos_arg))
    return per_construct, argmax, df


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=[os.path.join(OUT, "run_C0.json")])
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--report", default=os.path.join(OUT, "relevance_cosine_check.md"))
    ap.add_argument("--csv", default=os.path.join(OUT, "relevance_cosine_per_memory.csv"))
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(r).replace("run_", "").replace(".json", "")
                             for r in args.runs]

    embed = EmbeddingClient()
    cvecs = embed.embed([CONSTRUCT_PROMPTS[k] for k in CONSTRUCTS])

    out, per_memory = [], []
    p = out.append
    p("# Relevance validity — LLM judge vs cosine\n")
    p("Convergent-validity check: do the stored LLM-judge relevance scores track an "
      "independent cosine-to-construct signal? High = supportive; low = ambiguous "
      "(cosine is itself weak). NOT the multi-LLM IRR.\n")

    summary = {}
    for path, lab in zip(args.runs, labels):
        df = collect_memories(path)
        cos = cosine_relevance(df["text"], embed, cvecs)
        for i, k in enumerate(CONSTRUCTS):
            df[f"cos_{k}"] = cos[:, i]
        per_construct, argmax, df = evaluate(df)
        summary[lab] = argmax

        p(f"## {lab}  (n={len(df)} memories)\n")
        p("| construct | Spearman | Pearson | mean|diff| |\n|---|---|---|---|")
        for k in CONSTRUCTS:
            c = per_construct[k]
            p(f"| {k} | {c['spearman']:.3f} | {c['pearson']:.3f} | {c['mean_abs_diff']:.3f} |")
        p(f"\nArgmax (dominant construct) agreement: **{argmax['accuracy']:.1%}** "
          f"(Cohen's kappa {argmax['kappa']:.3f}); mean Spearman "
          f"{argmax['mean_spearman']:.3f}.\n")
        df.insert(0, "run", lab)
        per_memory.append(df)

    if len(summary) > 1:
        best = max(summary, key=lambda l: summary[l]["mean_spearman"])
        p(f"## Which judge aligns better with cosine\n")
        p("| run | mean Spearman | argmax acc | kappa |\n|---|---|---|---|")
        for lab, s in summary.items():
            p(f"| {lab} | {s['mean_spearman']:.3f} | {s['accuracy']:.1%} | {s['kappa']:.3f} |")
        p(f"\n→ **{best}** shows the higher cosine convergence (one weak signal among several; "
          "confirm with the multi-LLM IRR before concluding).\n")

    report = "\n".join(out)
    print(report)
    os.makedirs(OUT, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    pd.concat(per_memory, ignore_index=True).drop(columns="text").to_csv(
        args.csv, index=False, encoding="utf-8")
    print(f"\nSaved report -> {args.report}\nSaved per-memory CSV -> {args.csv}")


if __name__ == "__main__":
    main()
