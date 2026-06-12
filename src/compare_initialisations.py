"""Compare two agent initialisation runs (Nemotron vs Llama).

The two JSON files share an identical schema; only the LLM-inferred fields
(financial_security_score/reasoning, relationship_status/reasoning) can
differ. This script reports field-level differences, agreement statistics
for the two inferred variables, and exports a per-agent comparison CSV
to outputs/initialisation_comparison.csv for traceability.

Usage:
    python src/compare_initialisations.py
    python src/compare_initialisations.py --nemotron a.json --llama b.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def load_agents(path):
    with open(path, encoding="utf-8") as f:
        agents = json.load(f)
    return {a["agent_id"]: a for a in agents}


def field_diff_summary(nemotron, llama, agent_ids):
    fields = sorted(nemotron[agent_ids[0]].keys())
    rows = []
    for field in fields:
        n_diff = sum(
            json.dumps(nemotron[aid][field], sort_keys=True)
            != json.dumps(llama[aid][field], sort_keys=True)
            for aid in agent_ids
        )
        rows.append({"field": field, "agents_differing": n_diff})
    return pd.DataFrame(rows)


def build_comparison_df(nemotron, llama, agent_ids):
    rows = []
    for aid in agent_ids:
        a, b = nemotron[aid], llama[aid]
        rows.append({
            "agent_id": aid,
            "fin_score_nemotron": a["financial_security_score"],
            "fin_score_llama": b["financial_security_score"],
            "fin_delta": b["financial_security_score"] - a["financial_security_score"],
            "rel_status_nemotron": a["relationship_status"],
            "rel_status_llama": b["relationship_status"],
            "rel_status_source": a["relationship_status_source"],
            "rel_status_match": a["relationship_status"] == b["relationship_status"],
            "fin_reasoning_nemotron": a["financial_security_reasoning"],
            "fin_reasoning_llama": b["financial_security_reasoning"],
            "rel_reasoning_nemotron": a["relationship_status_reasoning"],
            "rel_reasoning_llama": b["relationship_status_reasoning"],
        })
    return pd.DataFrame(rows)


def report(df, summary):
    n = len(df)

    print("=" * 70)
    print("FIELD-LEVEL DIFFERENCES (count of agents where field differs)")
    print("=" * 70)
    print(summary.to_string(index=False))

    print()
    print("=" * 70)
    print(f"FINANCIAL SECURITY SCORE (n={n})")
    print("=" * 70)
    agree = (df["fin_delta"] == 0).sum()
    kappa = cohen_kappa_score(
        df["fin_score_nemotron"], df["fin_score_llama"], weights="quadratic"
    )
    print(f"Exact agreement: {agree}/{n} ({agree / n:.1%})")
    print(f"Quadratic-weighted Cohen's kappa: {kappa:.3f}")
    print(f"Mean score — nemotron: {df['fin_score_nemotron'].mean():.2f}, "
          f"llama: {df['fin_score_llama'].mean():.2f}")
    print("\nDelta distribution (llama - nemotron):")
    print(df["fin_delta"].value_counts().sort_index().rename("agents").to_string())
    print("\nConfusion matrix (rows = nemotron, cols = llama):")
    print(pd.crosstab(df["fin_score_nemotron"], df["fin_score_llama"],
                      rownames=["nemotron"], colnames=["llama"]).to_string())

    fin_flips = df[df["fin_delta"] != 0].sort_values(
        ["fin_delta", "agent_id"], key=lambda s: s.abs() if s.name == "fin_delta" else s,
        ascending=[False, True]
    )
    print(f"\nDisagreements ({len(fin_flips)}), largest deltas first:")
    for _, row in fin_flips.iterrows():
        print(f"\n{row['agent_id']}: nemotron={row['fin_score_nemotron']}, "
              f"llama={row['fin_score_llama']} (delta {row['fin_delta']:+d})")
        print(f"  nemotron reasoning: {row['fin_reasoning_nemotron']}")
        print(f"  llama reasoning:    {row['fin_reasoning_llama']}")

    print()
    print("=" * 70)
    print("RELATIONSHIP STATUS")
    print("=" * 70)
    imputed = df[df["rel_status_source"] == "llm_imputed"]
    agree_rel = imputed["rel_status_match"].sum()
    print(f"LLM-imputed agents: {len(imputed)} "
          f"(remaining {n - len(imputed)} come from raw marital status and cannot differ)")
    print(f"Agreement among imputed: {agree_rel}/{len(imputed)} "
          f"({agree_rel / len(imputed):.1%})")
    flips = imputed[~imputed["rel_status_match"]]
    if flips.empty:
        print("No disagreements.")
    else:
        print(f"\nDisagreements ({len(flips)}):")
        for _, row in flips.iterrows():
            print(f"\n{row['agent_id']}: nemotron={row['rel_status_nemotron']}, "
                  f"llama={row['rel_status_llama']}")
            print(f"  nemotron reasoning: {row['rel_reasoning_nemotron']}")
            print(f"  llama reasoning:    {row['rel_reasoning_llama']}")


def main():
    # Windows consoles default to cp1252, which chokes on characters in the
    # LLM reasoning text (e.g. non-breaking hyphens).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nemotron", default="agents_initialised.json",
                        help="Nemotron-initialised agents JSON")
    parser.add_argument("--llama", default="agents_llama_initialised.json",
                        help="Llama-initialised agents JSON")
    parser.add_argument("--out", default="outputs/initialisation_comparison.csv",
                        help="Per-agent comparison CSV")
    args = parser.parse_args()

    nemotron = load_agents(args.nemotron)
    llama = load_agents(args.llama)
    assert set(nemotron) == set(llama), "agent_id sets differ between files"
    agent_ids = sorted(nemotron)

    summary = field_diff_summary(nemotron, llama, agent_ids)
    df = build_comparison_df(nemotron, llama, agent_ids)
    report(df, summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved per-agent comparison to {out_path}")


if __name__ == "__main__":
    main()
