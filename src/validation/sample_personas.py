"""Draw the 500-persona validation sample from Nemotron-Personas-Singapore.

Stratified sample using the same filtering and M&P 2021 gender x marital
proportions as the agent pipeline in `fark_agent llama.ipynb`, scaled x2.5.
No duplicate rows within the sample (sampling is without replacement over
disjoint strata). Keeps the dataset `uuid` for traceability.

Note: same-seed pandas samples are nested, so this 500 contains the 200
production agents as a subset (same pool, seed 42). That is intentional —
the validation panel then also yields consensus labels for the production
agents directly.

From src/:  python validation/sample_personas.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import load_dataset

RANDOM_SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "..", "outputs", "validation",
                           "validation_personas_500.csv")

# Same mapping as the agent pipeline notebook, plus uuid for traceability.
FIELD_MAP = {
    "uuid":           "uuid",
    "age":            "age",
    "gender":         "sex",
    "rel_status":     "marital_status",
    "education":      "education_level",
    "occupation":     "occupation",
    "industry":       "industry",
    "planning_area":  "planning_area",
    "persona":        "general_persona",
    "cultural_bg":    "cultural_background",
    "hobbies":        "hobbies_and_interests",
    "career_goals":   "career_goals_and_ambitions",
}

# Agent-pipeline targets (50/47/46/57) scaled x2.5, rounded to sum 500.
TARGETS = {
    ("male",   "single"):  125,
    ("female", "single"):  118,
    ("male",   "married"): 115,
    ("female", "married"): 142,
}


def stratified_sample(df, targets, seed):
    """Sample `targets[(gender, rel)]` rows from each gender×marital cell (no replacement).

    Each cell is sampled independently to hit its exact target, so the combined
    sample matches the M&P gender×marital proportions. Raises if any cell's pool
    is too small to meet its target.
    """
    frames = []
    for (gender, rel), n in targets.items():
        cell = df[(df["gender"] == gender) & (df["rel_status"] == rel)]
        if len(cell) < n:
            raise RuntimeError(f"({gender}, {rel}): needed {n}, pool has {len(cell)}")
        frames.append(cell.sample(n, random_state=seed))
    return pd.concat(frames)


def main():
    assert sum(TARGETS.values()) == 500

    ds = load_dataset("nvidia/Nemotron-Personas-Singapore")
    df = pd.DataFrame(ds["train"])
    # FIELD_MAP is {our_name: dataset_name}; pandas rename wants {old: new}, so we
    # invert it to {dataset_name: our_name}. Then keep only our renamed columns.
    df = df.rename(columns={v: k for k, v in FIELD_MAP.items()})
    df = df[list(FIELD_MAP)].copy()

    # Same filters as the agent pipeline
    df = df[(df["age"] >= 21) & (df["age"] <= 45)].copy()
    df["gender"] = df["gender"].str.lower().str.strip()
    df = df[df["gender"].isin(["male", "female"])].copy()
    df["rel_status"] = df["rel_status"].str.lower().str.strip()
    df = df[df["rel_status"].isin(["single", "married"])].copy()
    print(f"Filtered pool: {len(df)} records")

    personas = stratified_sample(df, TARGETS, RANDOM_SEED)
    # Shuffle so any head-N subset (pilot runs) is demographically mixed
    # rather than stratum-ordered. Note: added 2026-06-12, which re-mapped
    # the V-ids relative to the first (unshuffled) CSV.
    personas = personas.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    personas.insert(0, "persona_id", [f"V{i:03d}" for i in range(1, len(personas) + 1)])

    assert len(personas) == 500
    assert personas["uuid"].is_unique, "duplicate personas in sample"

    out = os.path.abspath(DEFAULT_OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    personas.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(personas)} personas to {out}")
    print(personas.groupby(["gender", "rel_status"]).size().to_string())


if __name__ == "__main__":
    main()
