"""Compare prompt versions (v1 vs v2) on the pilot predictions.

Loads every outputs/validation/preds_<model>_<version>.jsonl, restricts to
personas rated under BOTH versions, and prints per-version agreement and
distribution metrics so the user can decide whether v2 replaces v1.

From src/:  python validation/pilot_report.py
"""

import glob
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "outputs", "validation")
PERSONAS_CSV = os.path.join(OUT_DIR, "validation_personas_500.csv")


def load_preds(out_dir):
    """All prediction records, deduped (last line per persona wins), no errors."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "preds_*_v*.jsonl"))):
        records = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records[rec["persona_id"]] = rec
        rows.extend(r for r in records.values() if not r.get("error"))
    return pd.DataFrame(rows)


def alpha_or_none(pivot, level):
    """Krippendorff alpha for a raters-x-personas pivot, None if undefined."""
    if pivot.shape[0] < 2:
        return None
    try:
        return krippendorff.alpha(reliability_data=pivot.to_numpy(dtype=float),
                                  level_of_measurement=level)
    except ValueError:
        return None


def mean_pairwise_kappa(pivot, weights=None):
    """Average Cohen's kappa over every pair of raters (rows of `pivot`).

    `weights="quadratic"` is used for the ordinal fin score so disagreeing by 3
    points is penalised far more than by 1. For each rater pair we keep only the
    personas both rated (`dropna`) and skip the pair if either rater gave the
    same score to everything (`nunique() > 1`) — kappa is undefined with zero
    variance.
    """
    kappas = []
    for a, b in combinations(pivot.index, 2):
        pair = pivot.loc[[a, b]].dropna(axis=1)
        if pair.shape[1] and pair.loc[a].nunique() > 1 and pair.loc[b].nunique() > 1:
            kappas.append(cohen_kappa_score(pair.loc[a], pair.loc[b],
                                            weights=weights))
    return float(np.mean(kappas)) if kappas else None


def report_version(df, version, personas):
    sub = df[df["prompt_version"] == version]
    print(f"\n{'=' * 60}\nPROMPT {version} — {sub['model_tag'].nunique()} model(s), "
          f"{sub['persona_id'].nunique()} personas\n{'=' * 60}")

    # --- Financial security score ---
    print("\nFinancial security score")
    stats = sub.groupby("model_tag")["fin_score"].agg(["mean", "std", "count"])
    dist = sub.groupby("model_tag")["fin_score"].value_counts().unstack(fill_value=0)
    print(stats.round(2).to_string())
    print("\nScore distribution:")
    print(dist.to_string())

    # Pivot to raters (rows) × personas (cols); both agreement metrics expect
    # one row per rater. fin_score is treated as ORDINAL (1-5 has a natural order).
    fin_pivot = sub.pivot_table(index="model_tag", columns="persona_id",
                                values="fin_score")
    alpha = alpha_or_none(fin_pivot, "ordinal")
    kappa = mean_pairwise_kappa(fin_pivot, weights="quadratic")
    alpha_s = f"{alpha:.3f}" if alpha is not None else "n/a (<2 raters or no variation)"
    print(f"\nKrippendorff alpha (ordinal): {alpha_s}")
    if kappa is not None:
        print(f"Mean pairwise quadratic kappa: {kappa:.3f}")

    # --- Relationship status (singles only) ---
    rel = sub[sub["rel_status"].isin(["Single", "Dating"])]
    print("\nRelationship status (singles)")
    if rel.empty:
        print("  no single personas rated yet")
        return
    share = rel.groupby("model_tag")["rel_status"].apply(
        lambda s: (s == "Dating").mean())
    print("% Dating per model:")
    print((share * 100).round(1).to_string())

    # rel_status is NOMINAL (Single vs Dating, no order), so we encode it 0/1 and
    # ask Krippendorff for nominal-level agreement.
    rel_pivot = rel.pivot_table(index="model_tag", columns="persona_id",
                                values="rel_status", aggfunc="first")
    codes = rel_pivot.map({"Single": 0, "Dating": 1}.get)
    alpha = alpha_or_none(codes, "nominal")
    alpha_s = f"{alpha:.3f}" if alpha is not None else "n/a (<2 raters or no variation)"
    print(f"Krippendorff alpha (nominal): {alpha_s}")
    if rel_pivot.shape[0] >= 2:
        unanimous = (rel_pivot.nunique() == 1).mean()
        print(f"% unanimous personas: {unanimous * 100:.1f}%")

    # --- Worst fin disagreements with reasoning ---
    if fin_pivot.shape[0] >= 2:
        spread = (fin_pivot.max() - fin_pivot.min()).sort_values(ascending=False)
        worst = spread[spread > 0].head(3)
        if len(worst):
            print("\nWorst fin disagreements:")
            for pid in worst.index:
                p = personas.loc[pid] if pid in personas.index else None
                desc = (f"{p['age']}yo {p['gender']} {p['occupation']}"
                        if p is not None else "")
                print(f"\n  {pid} (spread {int(worst[pid])}) {desc}")
                for _, r in sub[sub["persona_id"] == pid].iterrows():
                    print(f"    [{r['model_tag']}] {r['fin_score']}: "
                          f"{r['fin_reasoning']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = load_preds(OUT_DIR)
    if df.empty:
        print("No versioned prediction files found in outputs/validation.")
        return

    personas = pd.read_csv(PERSONAS_CSV).set_index("persona_id")

    versions = sorted(df["prompt_version"].unique())
    print(f"Loaded {len(df)} predictions: "
          + ", ".join(f"{m}/{v} n={len(g)}" for (m, v), g
                      in df.groupby(["model_tag", "prompt_version"])))

    # Fair comparison: only personas rated under every version present, so v1 and
    # v2 are scored on the exact same set (intersection of persona_ids per version).
    if len(versions) > 1:
        common = set.intersection(*(set(df[df["prompt_version"] == v]["persona_id"])
                                    for v in versions))
        print(f"Restricting to {len(common)} personas rated under all of "
              f"{versions}")
        df = df[df["persona_id"].isin(common)]

    for v in versions:
        report_version(df, v, personas)


if __name__ == "__main__":
    main()
