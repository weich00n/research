"""Build the final 100-agent pool from the validated v2 consensus labels.

This supersedes the old "regenerate 200 production agents" Phase D. Instead of
re-running LLM inference, it assembles a clean, fully-traceable 100-agent pool
directly from the completed 5-rater validation study:

  * financial_security_score = the 5-rater UNANIMOUS consensus value
    (only personas with fin_spread == 0 are eligible -> scores span 2/3/4).
  * financial_security_reasoning = gpt-4o-mini's reasoning text (strongest
    panel model, per user).
  * relationship_status = seed-42 random assignment (the LLM panel could not
    distinguish Dating from Single from persona text -- validation finding).

Sampling is stratified on gender x marital to the scaled M&P 2021 targets (the
only dimension the original pipeline hard-enforced); age and education are
reported as reference deltas (the original pipeline did not fit them either).

Reproducible: a single numpy default_rng(RANDOM_STATE) drives both the
stratified draw and the Single/Dating split, in a fixed, documented order.

Run:  .\.venv\Scripts\python.exe src\build_final_agents.py
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VAL = os.path.join(ROOT, "outputs", "validation")

CONSENSUS_CSV = os.path.join(VAL, "consensus_labels.csv")
PERSONAS_CSV = os.path.join(VAL, "validation_personas_500.csv")
GPT4O_JSONL = os.path.join(VAL, "preds", "gpt-4o-mini", "preds_gpt-4o-mini_v2.jsonl")
OUT_JSON = os.path.join(ROOT, "agents_final_100.json")

RANDOM_STATE = 42

# Scaled M&P 2021 gender x marital targets (sum 100 -> 49 Single / 51 Married,
# Male 48 / Female 52). These are the only dimension the original pipeline
# (history/fark_agent.ipynb, TARGETS dict) hard-enforced.
CELL_TARGETS = {
    ("male", "single"): 25,
    ("female", "single"): 24,
    ("male", "married"): 23,
    ("female", "married"): 28,
}
N_DATING = 24  # of the 49 singles; remaining 25 stay Single (CLAUDE.md ~25/~24)

# M&P reference distributions (NOT fitted by the original pipeline either) ──────
AGE_REF = {
    "single": {"21-25": 30, "26-30": 34, "31-35": 18, "36-40": 10, "41-45": 9},
    "married": {"21-25": 1, "26-30": 9, "31-35": 24, "36-40": 30, "41-45": 37},
}
EDU_REF = {
    "single": {"Secondary and below": 12, "Diploma / A-Level": 34, "Degree and above": 55},
    "married": {"Secondary and below": 16, "Diploma / A-Level": 28, "Degree and above": 56},
}


def age_band(age):
    a = int(age)
    if a <= 25:
        return "21-25"
    if a <= 30:
        return "26-30"
    if a <= 35:
        return "31-35"
    if a <= 40:
        return "36-40"
    return "41-45"


def norm_industry(raw):
    """Non-working personas (homemaker/student/unemployed) have no industry in the
    source; render an explicit placeholder instead of a blank / pandas 'nan'."""
    if raw is None or str(raw).strip().lower() in ("", "nan", "none"):
        return "Not applicable"
    return raw


def edu_bucket(raw):
    """Raw Nemotron education -> 3 M&P buckets (report only; stored field keeps raw)."""
    if raw == "University":
        return "Degree and above"
    if raw in ("Polytechnic", "Other Diploma", "Post Secondary (Non-Tertiary)"):
        return "Diploma / A-Level"
    return "Secondary and below"


# ── Load sources ───────────────────────────────────────────────────────────────
def load_personas():
    with open(PERSONAS_CSV, encoding="utf-8") as f:
        return {r["persona_id"]: r for r in csv.DictReader(f)}


def load_consensus():
    with open(CONSENSUS_CSV, encoding="utf-8") as f:
        return {r["persona_id"]: r for r in csv.DictReader(f)}


def load_gpt4o_reasoning():
    """Last non-error gpt-4o-mini record per persona (resumable JSONL: last wins)."""
    reasoning = {}
    with open(GPT4O_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            reasoning[rec["persona_id"]] = rec.get("fin_reasoning")
    return reasoning


def main():
    personas = load_personas()
    consensus = load_consensus()
    gpt4o = load_gpt4o_reasoning()

    # 1. spread == 0 pool ------------------------------------------------------
    pool = [pid for pid, c in consensus.items() if c["fin_spread"] == "0"]
    pool_cells = defaultdict(list)
    for pid in pool:
        p = personas[pid]
        pool_cells[(p["gender"], p["rel_status"])].append(pid)
    print(f"spread==0 pool: {len(pool)} personas")
    for cell, members in sorted(pool_cells.items()):
        print(f"  available {cell}: {len(members)}  (need {CELL_TARGETS[cell]})")

    # 2. stratified draw (single rng, fixed cell order) ------------------------
    # The order of rng calls below is LOAD-BEARING for reproducibility: cells are
    # drawn in sorted order, each candidate list is sorted before shuffling, and
    # the SAME rng then drives the Single/Dating split (step 3). Change the order
    # and the seed-42 selection changes.
    rng = np.random.default_rng(RANDOM_STATE)
    selected = []
    for cell in sorted(CELL_TARGETS):  # deterministic order
        cands = sorted(pool_cells[cell])  # deterministic before shuffle
        need = CELL_TARGETS[cell]
        if len(cands) < need:
            raise RuntimeError(f"pool short for {cell}: have {len(cands)}, need {need}")
        rng.shuffle(cands)
        selected.extend(cands[:need])

    # 3. Single/Dating split (same rng, after the draw) ------------------------
    singles = sorted(pid for pid in selected if personas[pid]["rel_status"] == "single")
    rng.shuffle(singles)
    dating = set(singles[:N_DATING])  # remaining stay Single

    # 4. assemble agents (sorted by persona_id for stable agent_id) ------------
    agents = []
    for i, pid in enumerate(sorted(selected), start=1):
        p = personas[pid]
        c = consensus[pid]
        marital = p["rel_status"]  # 'single' | 'married'

        if marital == "married":
            rel_status = "Married"
            rel_source = "raw_marital_status"
            rel_reasoning = "Marital status taken directly from the source dataset (married)."
        else:
            rel_status = "Dating" if pid in dating else "Single"
            rel_source = "randomized_seed42"
            rel_reasoning = (
                "Single/Dating randomly assigned with RANDOM_STATE=42; the LLM "
                "validation panel could not reliably distinguish dating from single "
                "from persona text (validation study)."
            )

        fin_reasoning = gpt4o.get(pid)
        if not fin_reasoning:
            raise RuntimeError(f"missing gpt-4o-mini reasoning for {pid}")

        agents.append({
            "agent_id": f"agent_{i:03d}",
            "age": int(p["age"]),
            "gender": p["gender"].capitalize(),          # Male / Female
            "marital_status": marital.capitalize(),       # Single / Married
            "relationship_status": rel_status,
            "relationship_status_source": rel_source,
            "relationship_status_reasoning": rel_reasoning,
            "education": p["education"],                   # raw Nemotron string
            "occupation": p["occupation"],
            "industry": norm_industry(p["industry"]),
            "planning_area": p["planning_area"],
            # CSV stores the consensus as text like "3.0"; float() then int()
            # parses it safely (int("3.0") would raise).
            "financial_security_score": int(float(c["fin_consensus_median"])),
            "financial_security_reasoning": fin_reasoning,
            "general_persona": p["persona"],
            "cultural_background": p["cultural_bg"],
            "hobbies_and_interests": p["hobbies"],
            "career_goals": p["career_goals"],
            "belief_state": {
                "attitude_score": 3,
                "subjective_norm_score": 3,
                "pbc_score": 3,
                "fertility_intention_dist": None,
            },
            "memory_stream": [],
            # provenance (ignored by sandbox.agent.Agent; for traceability only)
            "source_persona_id": pid,
            "uuid": p["uuid"],
            "fin_consensus_n_raters": int(c["n_raters_fin"]),
            "fin_consensus_spread": int(c["fin_spread"]),
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(agents, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(agents)} agents -> {os.path.relpath(OUT_JSON, ROOT)}")

    report(agents)


# ── Verification / M&P benchmark ───────────────────────────────────────────────
def _line(label, got, ref):
    g = f"{got:.0f}" if isinstance(got, float) else str(got)
    return f"    {label:<22} got {g:>5}   M&P {ref}"


def report(agents):
    n = len(agents)
    print("\n=== Verification summary ===")
    print(f"total agents: {n}")

    empties = [
        (a["agent_id"], k) for a in agents for k, v in a.items()
        if v in (None, "") and k != "fertility_intention_dist"
        and not (k == "belief_state")
    ]
    # belief_state.fertility_intention_dist is intentionally None
    print(f"empty required fields: {len(empties)}" + (f" -> {empties[:5]}" if empties else ""))

    # Gender x marital (ENFORCED target) -------------------------------------
    print("\n-- Gender x marital (enforced target, should match exactly) --")
    ct = Counter((a["gender"], a["marital_status"]) for a in agents)
    for (g, m), tgt in [(("Male", "Single"), 25), (("Female", "Single"), 24),
                        (("Male", "Married"), 23), (("Female", "Married"), 28)]:
        flag = "OK" if ct[(g, m)] == tgt else "!!"
        print(f"    {g}/{m:<8} got {ct[(g, m)]:>3}   target {tgt:>3}  [{flag}]")

    print("\n-- Relationship status --")
    for k, v in sorted(Counter(a["relationship_status"] for a in agents).items()):
        print(f"    {k:<10} {v}")
    print("-- Marital status --")
    for k, v in sorted(Counter(a["marital_status"] for a in agents).items()):
        print(f"    {k:<10} {v}")

    # Financial scores --------------------------------------------------------
    print("\n-- Financial security score (expect only 2/3/4) --")
    for k, v in sorted(Counter(a["financial_security_score"] for a in agents).items()):
        print(f"    score {k}: {v}")

    # Age band by marital (REFERENCE) ----------------------------------------
    print("\n-- Age band by marital (M&P reference, not fitted) --")
    for marital_title, marital_key in [("Single", "single"), ("Married", "married")]:
        grp = [a for a in agents if a["marital_status"] == marital_title]
        bands = Counter(age_band(a["age"]) for a in grp)
        print(f"  {marital_title} (n={len(grp)}):")
        for band in ["21-25", "26-30", "31-35", "36-40", "41-45"]:
            pct = 100 * bands[band] / len(grp) if grp else 0
            print(_line(band, pct, f"{AGE_REF[marital_key][band]}%"))

    # Education by marital (REFERENCE) ---------------------------------------
    print("\n-- Education (3-bucket) by marital (M&P reference, not fitted) --")
    for marital_title, marital_key in [("Single", "single"), ("Married", "married")]:
        grp = [a for a in agents if a["marital_status"] == marital_title]
        buckets = Counter(edu_bucket(a["education"]) for a in grp)
        print(f"  {marital_title} (n={len(grp)}):")
        for b in ["Secondary and below", "Diploma / A-Level", "Degree and above"]:
            pct = 100 * buckets[b] / len(grp) if grp else 0
            print(_line(b, pct, f"{EDU_REF[marital_key][b]}%"))


if __name__ == "__main__":
    main()
