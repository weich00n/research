  # News corpus workstream — consolidated results

**Date:** 2026-07-14 · **Model:** Qwen2.5-14B-Instruct (local vLLM)
**Companion docs:** `news_dissemination_design.md` (design + VacSim grounding),
`news_sanity_eval.md` (full per-article table), `network_area_decision.md`
(network side). Raw runs: `outputs/runs/smoke/c2_smoke_corpus.json`,
`outputs/runs/sanity/`.

## 1. What was built

C2/C3 policy news now comes from a **pre-generated factual article corpus**
(VacSim-style: variety instead of verbatim weekly repetition) instead of
re-announcing `policy.py` text. 8 policies × ~6 articles each (announcement /
explainer / family_impact / roundup), facts pinned to the Made For Families
M&P booklet (Apr 2025) + Budget 2025; served one per week, announcement
first, no text repeated within a run (`generate_news_corpus.py`,
`build_news_schedule(corpus_path=...)`, `driver.py --news-corpus`).

**Generation audit (3 rounds + curation):** every numeric token in every
article diffed against its fact block, flagged articles hand-checked.
Defects fell 7 → 3 → 3 as arithmetic rules and then *precomputed totals*
were added to the prompts; the 3 residual articles (multi-component total
errors) were removed by documented curation. **Final corpus: 45 articles**
(`news_corpus_qwen.json`, curation block embedded). Lesson: the LLM copies
provided figures reliably but *combines* them unreliably — give it every
total it might need.

## 2. Ratchet re-smoke (C2, 100 agents, 12 weeks)

Share of agents ≥4.8 (ceiling saturation) at week 12:

| construct | original C2 (broken) | reminder fix | **corpus** | policies' expected pathway? |
|---|---|---|---|---|
| attitude | 56% | 22% | **34%** | yes (3 caregiving policies) |
| subjective norm | 67% | 35% | **17%** | **no** (no policy targets norm) |
| pbc | 33% | 6% | **11%** | yes (all 8 policies) |

- Total saturation ≈ unchanged vs the reminder fix (62% vs 63% summed), but
  **redistributed onto the hypothesised constructs**: the reminder framing's
  worst saturation was on norm — the one construct no policy targets
  (weekly "the Government reminds you" reads as institutional pressure);
  the corpus moves attitude/pbc instead. Better for the specificity test.
- Mean E[intention] 2.96 → 3.09 (reminder fix: → 3.11). Monotonicity still
  ~95–100% in both — scores essentially never revert (stated limitation).
- Attitude saturation accelerates in weeks 9–12 and has not plateaued —
  **do not extend runs beyond 12 weeks** without revisiting.
- Decision: **corpus accepted** for C2/C3 (user + advisor aligned).

## 3. News sanity eval (per-article isolation, 45 cells + control)

Each article shown once to a fresh copy of the same 20 seeded agents,
2 update cycles, nothing else on; deltas net of a no-news control cell
(VacSim eval-mode-4 analogue). Full table: `news_sanity_eval.md`.

**(a) No article type is "hot" — corpus needs no surgery.**

| type | n | Δatt | Δnorm | Δpbc |
|---|---|---|---|---|
| announcement | 8 | +0.11 | +0.05 | +0.22 |
| explainer | 16 | +0.10 | +0.04 | +0.21 |
| family_impact | 14 | +0.13 | +0.05 | +0.21 |
| roundup | 7 | +0.10 | +0.04 | +0.22 |

The smoke's late-run attitude climb is therefore **accumulation** (every
article nudges +0.1–0.3 up, nothing pushes down), not any article type.

**(b) Control drift — the case for update gating.** With ZERO input,
agents drift **attitude +0.11 / pbc +0.06 in 2 weeks** (≈ +0.05/week
attitude) — the same order as a real article's effect. The forced weekly
update call manufactures positive drift from re-reading old memories.

**(c) Stimulus-level specificity: excellent.** 43/45 articles (and 8/8
policies pooled) move **pbc** hardest, exactly as hypothesised; norm ≈ +0.02
everywhere it should be. Financial → pbc confirmed at the single-article
level before any full run.

| policy | Δatt | Δnorm | Δpbc | expected |
|---|---|---|---|---|
| Baby Bonus & CDA | +0.16 | +0.02 | +0.21 | pbc ✓ |
| Child LifeSG Credits | +0.09 | +0.01 | +0.14 | pbc ✓ |
| Enhanced Paternity Leave | +0.11 | **+0.13** | +0.14 | pbc/att (norm!) |
| FWA Guidelines | +0.04 | +0.04 | +0.26 | pbc/att (att quiet) |
| Infant Childminding Pilot | +0.12 | +0.02 | +0.27 | pbc ✓ |
| Large Family Scheme | +0.14 | +0.02 | +0.19 | pbc ✓ |
| Preschool & Infant Care Subsidies | +0.14 | +0.03 | +0.30 | pbc ✓ |
| Shared Parental Leave | +0.12 | +0.07 | +0.21 | pbc/att ✓ |

**(d) Findings worth reporting, not fixing:**
- **Enhanced Paternity Leave is the only policy that moves norm**
  (+0.13 vs ~+0.02 elsewhere; its only 2 "top construct ≠ expected" rows are
  norm edging pbc by 0.01–0.02). Theoretically coherent: a *mandatory*
  entitlement reads as changed social expectations of fathers —
  injunctive-norm content the hypothesis sheet missed.
- **FWA Guidelines** hypothesised pbc+attitude but attitude barely moves
  (+0.04): its effect is nearly pure pbc.
- ΔE[intention] ≈ 0 for single exposures while constructs move — mediation
  operates with a lag; no shortcutting at the stimulus level.

## 4. Where the saturation actually comes from (and the two fixes)

The sanity eval decomposes the smoke's drift into two measured mechanisms,
matching the advisor's diagnosis ("positive news would definitely lead to
those constructs increasing"):

1. **No-input drift** (+0.05/week attitude with nothing new) → fix:
   **update gating** — skip the belief-update LLM calls in weeks where an
   agent received no new information; carry scores forward.
2. **Uniformly positive environment** (all 45 articles push up; nothing
   pushes down) → fix: **context/"bad news" corpus** (cost-of-living,
   housing waits, job-market pressure — factual, theory-blind, not policy
   stances), enabling the *policy-effectiveness-under-different-situations*
   factorial: {policy on/off} × {negative context on/off}.

## 5. Status / next steps

- [x] Corpus generated, audited (3 rounds), curated (45 articles), smoke-run
- [x] Sanity harness (`validation/news_sanity_eval.py`) + full sweep
- [ ] Update gating in `engine.py` (+ re-smoke; measured justification in §3b)
- [ ] Context news corpus + 3-arm factorial (C2 / C2-headwind / context-only)
- [ ] Real matrix: C2 combined + financial + caregiving arms, C3 on
      `social_network_qwen_noarea.json`; then mediation analysis script
