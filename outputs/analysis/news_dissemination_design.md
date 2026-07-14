# News dissemination design: pre-generated factual policy corpus

**Date:** 2026-07-13 · **Status:** adopted (user-approved plan)
**Companion:** `network_area_decision.md` (network side of the C1/C3 inputs).

## The problem

C2/C3 broadcast one policy news item per week by cycling the eight
`src/sandbox/policy.py` instruments verbatim. Re-announcing identical text
weekly read as fresh evidence each time and ratcheted TPB scores monotonically
toward the ceiling. The interim fix (habituation prompts + "reminder" framing
on repeat cycles, smoke-validated 2026-07-03) treats the symptom in framing;
the stimulus itself is still the same sentence every cycle.

## What VacSim actually does (arXiv 2503.09639)

VacSim separates two information channels:

1. **News network (§3.2).** A pre-generated corpus of ~10,000 LLM-written
   articles (Llama-3.1-8B, temperature 1.5, 20 real articles as in-context
   style examples), each ~250 tokens, in four stance types (pro/anti-vaccine,
   life-disrupted/life-unaffected). Each timestep a personalized recommender
   scores candidate articles by max cosine similarity against the agent's own
   past tweets and serves the top K=3 — a heterogeneous, endogenous media diet.
   No article is ever repeated.
2. **Policy channel (§3.1, §3.3).** The intervention (incentive / ambassador /
   mandate) is delivered as **uniform factual text broadcast identically to
   all agents**, separate from news. No stance engineering.

**Their validation of the information channel (§4.2, "Altering news
stances"):** run the simulation under all-positive vs all-negative news and
verify hesitancy moves in the hypothesised direction across models — a
directional sanity check, not a content benchmark. Supporting methodology
they build on: Park et al. (2023) generative agents; Törnberg et al. (2023)
LLM social-media simulation.

## Mapping to this project

Our stimuli **are policies**, so they map to VacSim's *policy channel*, not
its news channel. Three consequences, all user-confirmed:

- **Factual/neutral articles only.** VacSim's stance mixing belongs to its
  contested media channel; Singapore pro-natal policy coverage is
  predominantly factual reporting of schemes. Engineering negative coverage
  would redefine the C2/C3 treatment away from "policy exposure".
- **No personalized recommender.** Endogenous exposure (articles selected by
  similarity to an agent's own posts) would confound the mediation analysis —
  exposure would depend on the agent's evolving state. Uniform broadcast
  keeps the treatment exogenous, matching VacSim's policy channel.
- **Corpus scope = the 8 registered policies only** (each has an
  `expected_pathways` hypothesis in `policy.py`). Articles about untracked
  instruments (housing grants, IVF co-funding, tax reliefs) would be
  treatments the mediation analysis cannot attribute.

What we **do adopt** from the news channel is the anti-ratchet mechanism:
**variety instead of verbatim repetition**. A small pre-generated corpus of
distinct factual articles per policy (announcement, explainer, illustrative
family-impact, in-effect roundup) means an agent never reads the same text
twice in a run; repeat coverage of a policy arrives as *ongoing coverage*
(explainers, roundups), not re-announcement. The first appearance of each
policy in a run is always its announcement article; later appearances draw
only non-announcement types.

## Source material

Facts come from the official **Made For Families "Support for Your Marriage
& Parenthood Journey" booklet (Apr 2025)**
(https://www.madeforfamilies.gov.sg/docs/default-source/default-document-library/m-p-booklet-(apr-2025).pdf),
which covers 7 of the 8 policies with exact amounts, dates, and eligibility
(and worked family examples). The eighth — Child LifeSG Credits (ages 0–12)
— is a Budget 2025 / SG60 measure not detailed in the booklet; its fact block
comes from the Budget 2025 announcement. Per-policy fact blocks are embedded
in `src/generate_news_corpus.py`; generation prompts forbid inventing
statistics, uptake figures, officials, or quotes, and contain **no TPB
vocabulary** (theory-blind inputs). Journalistic style comes from two
handwritten neutral exemplars (about non-corpus topics, so style cannot leak
content) — the analogue of VacSim's real-article style few-shots.

## Corpus generation audit (2026-07-14)

Three generation rounds on Qwen2.5-14B-Instruct, each audited by number-diff
(every numeric token in every article checked against its fact block) plus a
manual arithmetic read of all flagged articles:

- **Round 1** (47/48 slots): 7 defective articles — invented out-of-pocket
  computations, fee caps misapplied across age groups / birth orders, a
  fabricated benefit amount, a one-off credit described as annual. All but
  one in the `family_impact` type (the one doing arithmetic).
- **Round 2** (48/48, after adding explicit arithmetic rules + a one-off
  clarification to the Child LifeSG fact block): 3 defects — all
  multi-component totals (CDA components, SPL phase boundary).
- **Round 3** (48/48, after embedding *precomputed* totals in the Baby Bonus
  and SPL fact blocks so the model never sums): SPL fully correct; 3 residual
  defects (a Baby Bonus variant fabricating S$38,000, a Baby Bonus roundup
  headline mixing birth orders, one confused preschool fee-cap sentence).
- **Curation:** the 3 residual articles were removed by documented human
  audit (see the `curation` block inside `news_corpus_qwen.json`; prior
  rounds preserved as `news_corpus_qwen_audit*.json`). Final corpus:
  **45 articles**, covering combined / financial-only / caregiving-only
  12-week schedules with no repeats and no fallback texts.

Lesson recorded: LLMs at generation temperature reliably copy provided
figures but unreliably *combine* them — fact blocks should carry every total
the article might need, and a number-diff audit is cheap and catches most of
what goes wrong.

## Validation plan (VacSim-analogous)

1. **Ratchet re-smoke (required).** The corpus replaces the reminder-news
   half of the validated ratchet package, so the C2 smoke (100 agents ×
   13 weeks, same config as `c2_smoke_ratchet_fix`) is rerun with the corpus.
   Pass = ceiling saturation (share of agents ≥4.8 per construct) at or below
   the validated fix's levels (att 22% / norm 35% / pbc 6% at week 12).
2. **Directional sanity check (§4.2 analogue).** Financial-policy news should
   move `pbc` more than the other constructs — inspected on the smoke via the
   run dashboard; the formal specificity/mediation test is the mediation
   analysis script.

## References

- Liu, S. et al. (2025). Can a society of generative agents simulate human
  behavior and inform public health policy? A case study on vaccine
  hesitancy. arXiv:2503.09639.
- Park, J. S. et al. (2023). Generative agents: Interactive simulacra of
  human behavior. *UIST 2023*.
- Törnberg, P. et al. (2023). Simulating social media using large language
  models to evaluate alternative news feed algorithms. arXiv:2310.05984.
- Made For Families (2025). Support for Your Marriage & Parenthood Journey
  (Apr 2025 booklet). Singapore: Made For Families / MSF.
