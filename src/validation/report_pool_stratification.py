"""Saved report: the 100-agent pool's demographic fit to M&P 2021.

`build_final_agents.report()` prints the same checks to stdout at build time but
saves nothing; this read-only script regenerates them from the existing
`agents_final_100.json` and writes a durable, citable report to `reports/`
(+ a per-cell CSV to `outputs/analysis/`). Nothing is mutated.

Reuses the build script's M&P targets and bucketing verbatim
(`CELL_TARGETS`, `AGE_REF`, `EDU_REF`, `age_band`, `edu_bucket`) so the report
can never drift from how the pool was actually drawn.

Run from src/:
    python validation/report_pool_stratification.py
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import date

# src/validation/ -> put src/ on the path so `build_final_agents` imports the
# same way it does for the build pipeline (run from repo root / src).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from build_final_agents import (  # noqa: E402  (path set above)
    AGE_REF, CELL_TARGETS, EDU_REF, age_band, edu_bucket,
)

DEFAULT_AGENTS = os.path.join(ROOT, "agents_final_100.json")
DEFAULT_REPORT = os.path.join(ROOT, "reports", "agent_pool_mp_stratification.md")
DEFAULT_CSV = os.path.join(ROOT, "outputs", "analysis", "baseline", "pool_stratification_cells.csv")

AGE_BANDS = ["21-25", "26-30", "31-35", "36-40", "41-45"]
EDU_BUCKETS = ["Secondary and below", "Diploma / A-Level", "Degree and above"]
# The enforced cells, in the same fixed order build_final_agents.report() uses.
ENFORCED_CELLS = [("Male", "Single"), ("Female", "Single"),
                  ("Male", "Married"), ("Female", "Married")]


def load_agents(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ref_rows(agents, marital_title, marital_key, key_fn, categories, ref):
    """Per-category got% vs M&P ref% (+ delta) for one marital group.

    Returns (rows, n) where each row is (label, got_pct, ref_pct, delta).
    """
    grp = [a for a in agents if a["marital_status"] == marital_title]
    counts = Counter(key_fn(a) for a in grp)
    n = len(grp)
    rows = []
    for cat in categories:
        got = 100 * counts[cat] / n if n else 0.0
        ref_pct = ref[marital_key][cat]
        rows.append((cat, got, ref_pct, got - ref_pct))
    return rows, n


def frame_diagnostic():
    """Eligible (fin-unanimous) pool age distribution for married personas.

    Evidences how much of the married-age deviation is the *frame* (the
    validated pool's own composition) vs the *draw* (within-cell random
    sampling). Returns None if the validation CSVs aren't present.
    """
    try:
        from build_final_agents import load_consensus, load_personas
        personas = load_personas()
        consensus = load_consensus()
    except (FileNotFoundError, OSError):
        return None
    married = [pid for pid, c in consensus.items()
               if c["fin_spread"] == "0" and personas[pid]["rel_status"] == "married"]
    if not married:
        return None
    counts = Counter(age_band(int(personas[pid]["age"])) for pid in married)
    n = len(married)
    return {"n": n,
            "pct": {b: 100 * counts[b] / n for b in AGE_BANDS},
            "count": {b: counts[b] for b in AGE_BANDS}}


def build_report(agents):
    n = len(agents)
    out = []
    p = out.append

    p("# Agent Pool — Demographic Stratification vs M&P 2021\n")
    p(f"**Date:** {date.today().isoformat()}  ")
    p(f"**Source:** `agents_final_100.json` ({n} agents, neutral beliefs)  ")
    p("**Builder:** `src/build_final_agents.py` (seed `RANDOM_STATE=42`)\n")
    p("> Stratified draw of 100 personas from the **financial-unanimous** "
      "validation pool (`fin_spread == 0`). **Gender × marital is the only "
      "enforced dimension**; age and education are reported as M&P *reference "
      "deltas* — they were never fitted, by design.\n")

    # ── 1. Gender × marital (enforced) ──────────────────────────────────────
    p("## 1. Gender × marital — enforced (must match exactly)\n")
    ct = Counter((a["gender"], a["marital_status"]) for a in agents)
    p("| cell | got | M&P target | |")
    p("|---|---|---|---|")
    all_ok = True
    for g, m in ENFORCED_CELLS:
        tgt = CELL_TARGETS[(g.lower(), m.lower())]
        ok = ct[(g, m)] == tgt
        all_ok = all_ok and ok
        p(f"| {g}/{m} | {ct[(g, m)]} | {tgt} | {'✅' if ok else '❌'} |")
    verdict = ("All four cells hit their scaled M&P targets exactly"
               if all_ok else "⚠️ One or more cells deviate from target")
    p(f"\n{verdict} → **{sum(1 for a in agents if a['marital_status'] == 'Single')} "
      f"Single / {sum(1 for a in agents if a['marital_status'] == 'Married')} "
      f"Married**, "
      f"**Male {sum(1 for a in agents if a['gender'] == 'Male')} / "
      f"Female {sum(1 for a in agents if a['gender'] == 'Female')}**.\n")

    # ── 2. Relationship + marital splits ────────────────────────────────────
    p("## 2. Relationship & marital status\n")
    p("Marital (raw M&P category) is enforced; the 49 singles are split into "
      "Single/Dating by seed 42 (the LLM panel could not recover Dating from "
      "persona text).\n")
    rel = Counter(a["relationship_status"] for a in agents)
    p("| relationship_status | n |")
    p("|---|---|")
    for k in ("Single", "Dating", "Married"):
        p(f"| {k} | {rel.get(k, 0)} |")
    p("")

    # ── 3. Financial-security score ─────────────────────────────────────────
    p("## 3. Financial-security score\n")
    p("LLM-inferred, stored as the **5-rater unanimous consensus** — so the "
      "range is intrinsically **2–4** (the panel never near-unanimously assigns "
      "1 or 5).\n")
    fin = Counter(a["financial_security_score"] for a in agents)
    mean_fin = sum(a["financial_security_score"] for a in agents) / n
    p("| score | n |")
    p("|---|---|")
    for s in sorted(fin):
        p(f"| {s} | {fin[s]} |")
    p(f"\nMean financial-security score: **{mean_fin:.2f}**.\n")

    # ── 4 & 5. Reference-delta tables (age, education) ──────────────────────
    csv_rows = []  # (dimension, group, category, got_pct, ref_pct, delta)
    for (g, m) in ENFORCED_CELLS:
        csv_rows.append(("gender_x_marital", "-", f"{g}/{m}",
                         ct[(g, m)], CELL_TARGETS[(g.lower(), m.lower())],
                         ct[(g, m)] - CELL_TARGETS[(g.lower(), m.lower())]))

    for title, (dim, key_fn, cats, ref) in {
        "## 4. Age band by marital — M&P reference (not fitted)\n":
            ("age_band", lambda a: age_band(a["age"]), AGE_BANDS, AGE_REF),
        "## 5. Education (3-bucket) by marital — M&P reference (not fitted)\n":
            ("education", lambda a: edu_bucket(a["education"]), EDU_BUCKETS, EDU_REF),
    }.items():
        p(title)
        for marital_title, marital_key in [("Single", "single"), ("Married", "married")]:
            rows, gn = _ref_rows(agents, marital_title, marital_key, key_fn, cats, ref)
            p(f"**{marital_title}** (n={gn})\n")
            p("| category | got % | M&P % | Δ |")
            p("|---|---|---|---|")
            for label, got, ref_pct, delta in rows:
                p(f"| {label} | {got:.0f} | {ref_pct} | {delta:+.0f} |")
                csv_rows.append((dim, marital_title, label,
                                 round(got, 1), ref_pct, round(delta, 1)))
            p("")

    # ── 6. justification: why age/education aren't fitted ───────────────────
    p("## 6. Why age & education are reported, not fitted\n")
    p("Only **gender × marital** is enforced; age and education are unweighted "
      "reference deltas — by design:\n")
    p("1. **The estimand is mechanism, not representativeness.** The study is a "
      "TPB mediation/specificity test *inside* the simulation (does a policy route "
      "through its hypothesised construct?) — a structural property of the agent "
      "architecture that does not require the age marginal to match Singapore. It "
      "is explicitly **not** a forecast of national intention levels or TFR.")
    p("2. **Gender × marital is the demographic axis that most directly gates "
      "fertility intention**, so it is the one enforced; age/education are "
      "secondary and partly collinear with marital. This also continues the "
      "original pipeline's single enforced dimension.")
    p("3. **The eligible frame cannot support more cells at n=100.** The draw is "
      "restricted to the **financial-unanimous** subset (~190 personas); stacking "
      "age (5 bands) × education (3) onto gender×marital would create ~120 cells "
      "and force empty ones.")
    p("4. **The largest deviation is benign for an *intention* study:** the "
      "married sample concentrates in 31–35 (the live childbearing-decision "
      "window) rather than 41–45 (largely completed fertility).\n")

    diag = frame_diagnostic()
    if diag:
        married_sel = [a for a in agents if a["marital_status"] == "Married"]
        sel = Counter(age_band(a["age"]) for a in married_sel)
        sel_pct = {b: 100 * sel[b] / len(married_sel) for b in AGE_BANDS}
        p("**The married-age gap is mostly the frame, not a fitting error.** The "
          "fin-unanimous eligible pool is itself younger than M&P married, so even "
          "a faithful draw inherits the skew:\n")
        p("| married age | eligible pool % | selected % | M&P % |")
        p("|---|---|---|---|")
        for b in AGE_BANDS:
            p(f"| {b} | {diag['pct'][b]:.0f} | {sel_pct[b]:.0f} | {AGE_REF['married'][b]} |")
        pool45, sel45 = diag["pct"]["41-45"], sel_pct["41-45"]
        mp45 = AGE_REF["married"]["41-45"]
        avail45 = diag["count"]["41-45"]
        need45 = round(mp45 / 100 * len(married_sel))
        p(f"\nFor 41–45, the {mp45 - sel45:.0f}pp gap splits into roughly "
          f"~{mp45 - pool45:.0f}pp **frame** (the validated pool holds only "
          f"{avail45} older-married personas = {pool45:.0f}%) and ~{pool45 - sel45:.0f}pp "
          f"**draw** (un-stratified within-cell sampling). Older-married personas "
          f"*were* available ({avail45} vs ~{need45} needed), so the skew is "
          f"**recoverable** by post-stratification weighting if an analysis ever "
          f"needs population levels.\n")

    # ── limitations + reproducibility ───────────────────────────────────────
    p("## Limitations\n")
    p("- **Scope of this calibration:** the lenient age/education fit is "
      "acceptable *because* the estimand is a within-model mechanism. If baseline "
      "results are ever compared to real Singapore intention **levels** or used to "
      "claim population representativeness, the married-age skew should be "
      "corrected by post-stratification weighting (older-married personas exist in "
      "the frame).")
    p("- Only **gender × marital** is fitted; age/education deltas are expected "
      "and **not** corrected. The pool is the eligible **fin-unanimous** subset, "
      "which constrains who could be drawn.")
    p("- Relationship Single/Dating is **randomised** (seed 42), not inferred — "
      "treat Dating as a low-confidence attribute.")
    p("- This is a *demographic* characterisation of the static pool; the "
      "initialised belief baseline is a separate report "
      "(`reports/c0_initialised_baseline.md`).\n")

    p("## Reproducibility / artifacts\n")
    p("- **Script:** `src/validation/report_pool_stratification.py` (read-only).")
    p("- **Per-cell CSV:** `outputs/analysis/pool_stratification_cells.csv`.")
    p("- **Targets/bucketing** imported verbatim from `src/build_final_agents.py` "
      "(`CELL_TARGETS`, `AGE_REF`, `EDU_REF`, `age_band`, `edu_bucket`).")
    p("- **Regenerate:** from `src/`, `python validation/report_pool_stratification.py`")

    return "\n".join(out), csv_rows


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()

    agents = load_agents(args.agents)
    report, csv_rows = build_report(agents)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "group", "category", "got", "ref", "delta"])
        w.writerows(csv_rows)

    print(report)
    print(f"\nSaved report -> {os.path.relpath(args.report, ROOT)}")
    print(f"Saved per-cell CSV -> {os.path.relpath(args.csv, ROOT)}")


if __name__ == "__main__":
    main()
