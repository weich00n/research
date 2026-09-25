# CLAUDE.md — Singapore Fertility Intention ABM

This file brings any AI agent (or new developer) up to speed on the project's purpose, design decisions, data pipeline, and coding conventions. Read this before touching any file.

---

## Research Question

> How do policy type, repeated exposure, ambient information, and social interaction jointly shape fertility intentions in an LLM-agent population?

*(Supervisor-approved 2026-09-25. Supersedes the earlier TPB-mediation framing, retained below as a secondary robustness dimension.)*

The outcome variable is **fertility intention alone**. The primary contrast is **policy type**: does a sustained stream of financial policy news move intention differently from a sustained stream of caregiving policy news? Four factors:

| Factor | Levels |
|---|---|
| Policy type | financial / caregiving |
| Repeated exposure | single article / 12 weeks sustained |
| Ambient information | policy-only / mixed-valence context |
| Social interaction | off (C2) / on (C3) |
| *Agent architecture (robustness)* | *explicit belief layer / none* |

This is an **agent-based model (ABM)**, not a predictive demographic model. The goal is to study *mechanisms*, not to forecast Singapore's actual TFR.

**Why TPB is no longer the lead framing.** The mediation test was run and the data did not support it. Under policy input the constructs do not mediate intention: PBC moves ~1.7 points on the 1–5 scale while its correlation with intention change is ~0 (r = −0.013, n = 1,200 agent-weeks), and this survives a construct-blind retrieval ablation. The belief layer also produces a saturation artifact (95–100% of agents monotone non-decreasing under uniformly positive news) that the no-TPB engine does not. The one place mediation is real is the social-only condition, where it survives ablation. TPB is therefore retained as an **architecture-robustness dimension** — does the policy ranking hold with and without a belief layer — not as the object of study.

**Scope of claim (important).** Where TPB results are reported, they evaluate TPB's **fidelity as an agent architecture for LLM-based simulation**, **not** whether TPB is empirically true of real humans. The LLM has been trained on TPB and the construct labels appear in the prompts, so the simulation cannot adjudicate the theory's validity for human fertility.

---
## Theoretical Backbone: Theory of Planned Behaviour (TPB)

Every agent's internal state is structured around three TPB constructs. Demographic attributes do **not** directly cause fertility intention — they shape beliefs, which shape intention.

| Construct | Meaning | Example drivers |
|---|---|---|
| `attitude_score` | Does the agent evaluate having a child positively or negatively? | Financial burden, emotional fulfilment, career trade-off |
| `subjective_norm_score` | Does the agent feel social pressure / perceived normalcy from important referents around having children? (injunctive = what referents think you should do; descriptive = what referents themselves are doing) | Parental expectations, peer childbearing, government messaging |
| `pbc_score` (Perceived Behavioural Control) | Does the agent feel *capable* of having and raising a child? | Income, housing readiness, childcare access, work-life balance |
| `fertility_intention_score` | Probability distribution over intention levels | Output variable — see below |

All TPB scores use a **1–5 scale**. Agents are *created* at neutral (`attitude=3, norm=3, pbc=3`), then a one-time deterministic **seed-only baseline pass (t=0)** sets grounded, profile-differentiated scores + intention from the seed memories (full 1–5 range, no "stay near 3" anchor). This baseline is frozen into `agents_final_100_seeded.json` (via `driver.py --init-only`) and shared by C0–C3, so every condition starts identical. The neutral 3/3/3 remains the scale's midpoint, not a starting belief.

---

## Fertility Intention Score
The dependent variable is a **probability distribution over 5 ordinal levels**, not a single number.

```
[p1, p2, p3, p4, p5]

1 = No child intention
2 = Weak / unlikely
3 = Uncertain
4 = Likely
5 = Strong intention
```

Example for a pro-fertility agent: `[0.0, 0.1, 0.1, 0.7, 0.1]`

Always output this as a list of 5 floats summing to 1.0.

---

## Agent Structure (3 Layers)

### Layer 1 — Static Profile (never changes during simulation)

Sourced from the **Nemotron-Personas-Singapore** dataset, sampled to match the **2021 Singapore Marriage & Parenthood (M&P) Survey** distributions. Field names below are the actual keys in `agents_final_100.json` read by `sandbox.agent.Agent`.

| Field | Type | Notes |
|---|---|---|
| `agent_id` | string | Format `agent_001`–`agent_100` (assigned in persona-id sort order) |
| `age` | int | Range: 21–45 |
| `gender` | string | `Male` / `Female` |
| `marital_status` | string | `Single` / `Married` (raw M&P category) |
| `relationship_status` | string | `Single` / `Dating` / `Married` (Dating split from Single by seed — see note) |
| `education` | string | Raw Nemotron level (`University`, `Polytechnic`, `Other Diploma`, `Post Secondary (Non-Tertiary)`, `Secondary`, `No Qualification`); the 3-bucket M&P mapping is used only for reporting |
| `occupation` | string | From Nemotron |
| `industry` | string | From Nemotron (`Not applicable` for non-working personas) |
| `planning_area` | string | Singapore planning area (residence) |
| `financial_security_score` | int | 1–5 scale; the unanimous-consensus pool only spans **2–4**. LLM-inferred — see note |
| `financial_security_reasoning` | string | gpt-4o-mini's reasoning for the inferred score |
| `general_persona` | string | Nemotron narrative persona |
| `cultural_background` | string | Narrative, not deterministic |
| `hobbies_and_interests` | string | Narrative, not deterministic |
| `career_goals` | string | Affects perceived parenthood–career trade-off |

Each agent also carries **provenance fields not consumed by `Agent`**: `relationship_status_source`, `relationship_status_reasoning`, `source_persona_id`, `uuid`, `fin_consensus_n_raters`, `fin_consensus_spread` (see the Final agent pool note under File Conventions).

**Note on `relationship_status`:** The M&P survey only reports `Single`/`Married`. Single agents are split 50/50 into `Single` and `Dating` by seed (`RANDOM_STATE=42`); the LLM validation panel could **not** recover `Dating` from persona text, so it is randomised rather than inferred.

**Note on `financial_security_score`:** Income is not in the dataset, so this is **LLM-inferred** (1 = Low … 5 = Upper). The stored value is the **5-rater unanimous consensus** from the validation study (so it spans only 2–4); `financial_security_reasoning` holds gpt-4o-mini's explanation.

### Layer 2 — Dynamic Belief State (updates each timestep)

```python
{
    "attitude_score": float,          # 1–5
    "subjective_norm_score": float,   # 1–5
    "pbc_score": float,               # 1–5
    "fertility_intention_dist": [float x5] | None  # distribution summing to 1.0; None until first update
}
```

### Layer 3 — Memory Stream (grows each timestep)

Each memory object:

```python
{
    "memory_id": str,
    "agent_id": str,
    "created_timestep": int,
    "memory_text": str,               # natural language
    "source_type": str,               # "profile_seed" | "policy_news" | "social_post" | "reflection" | "conversation"
    "source_agent_id": str | None,    # which agent caused this memory (if social)
    "importance": float,              # 0–1, not memorable → life-changing
    "available_from_timestep": int,   # posts visible to others only from t+1
    "memory_class": str,              # "seed" | "simulation"
    "relevance": dict,                # {attitude,norm,pbc} in [0,1]; what retrieval ranks by (LLM-judge scores in hybrid mode, filled lazily by rerank)
    "cosine_relevance": dict,         # {attitude,norm,pbc} cosine prefilter signal (hybrid mode only); shortlists rerank candidates, not used by retrieval
    "retrieval_history": list,        # when retrieved and for which TPB construct
    "used_in_update": bool
}
```

---

## Simulation Loop (1 timestep = 1 week)

**Before week 1 (once):** generate 5 seed memories per agent, then run a deterministic **seed-only baseline pass at t=0** that *sets* each agent's TPB scores + intention distribution from those seeds (full 1–5 range, no neutral anchor, scores only — no reflection memory). This is frozen into `agents_final_100_seeded.json` and shared across C0–C3. The weekly loop below then runs from t=1 against that grounded baseline.

Each timestep, for each agent:

1. Receive fertility-related **policy news** (if in policy condition).
2. Read **social posts** from connected agents (posted at t-1).
3. Convert inputs into new **fertility-relevant memories** (tagged by TPB construct).
4. Calculate **saliency** for all memories: `saliency = importance × λ^(current_t − created_t)` where `λ = 0.995`.
5. Calculate **TPB relevance** for each memory. Three switchable modes (`--relevance-mode`): `llm` (LLM-as-judge every memory at creation), `cosine` (embedding similarity to the construct embed anchors), or `hybrid` (**recommended for norm**: cheap cosine prefilter → LLM rerank — at retrieval, take the cosine top-K per construct, union them, and LLM-judge only that shortlist, cached per memory).
6. **Retrieve top memories**: up to 5 seed/profile memories + up to 5 per TPB construct from simulation memories (max 20 total, no duplicates).
7. Two **decoupled** LLM calls: (7a) **update TPB scores** (`attitude`, `subjective_norm`, `pbc`) + reflection memory, and (7b) **update the `fertility_intention` distribution** in a separate call that does **not** see the TPB scores — so the TPB→intention link is *measured* (mediation), not *instructed*.
8. Agent may **post** a fertility-related message to the social network (available to connected agents at t+1).
9. **Save** updated belief state, intention score, and any new post.

---

## Memory Retrieval Formula

```
retrieval_score = normalised_saliency × TPB_relevance_score
```

Retrieve top-5 per TPB construct. If a memory would appear in multiple construct retrievals, include it only once (no duplicates). Cap total retrieved memories at 20.

---

## Experimental Conditions

| Condition | Policy News | Social Posts | Purpose |
|---|---|---|---|
| C0 — Static Baseline | ✗ | ✗ | The t=0 baseline (seed-only), then frozen — control / drift floor |
| C1 — Social Only | ✗ | ✓ | Isolate peer/social influence |
| C2 — Policy Only | ✓ | ✗ | Isolate direct policy effect |
| C3 — Policy + Social | ✓ | ✓ | Test interaction / amplification |

### Policy Scenarios

All policies are grounded in real Singapore instruments:

**Financial:**
- Baby Bonus & Child Development Account
- Large Family Scheme (CDA top-up, MediSave grant, LifeSG credits)
- Child LifeSG Credits (ages 0–12)

**Caregiving:**
- Enhanced Paternity Leave (2 → 4 weeks; additional 2 weeks mandatory from April 2025)
- Shared Parental Leave
- Flexible Work Arrangement Request Guidelines
- Preschool & Infant Care subsidies
- Infant Childminding Pilot

**Expected TPB pathways:**
- Financial support → mainly `pbc_score`
- Parental leave / flexible work → `pbc_score` + `attitude_score`
- Combined policy → `pbc_score`, `attitude_score`, possibly `subjective_norm_score`

**NDR 2026 (opt-in, reported separately).** The National Day Rally of 17 Aug
2026 announced four further instruments, held in `policy.NDR2026_POLICIES`:
SG Child Support Package (financial; replaces Baby Bonus + Large Family
Scheme), Enhanced Childcare Leave (caregiving), Preschool & Infant Care Fee
Reduction (caregiving), and Additional BTO Ballot Chance for Families
(**housing** — a new category with no pre-2026 member).

The eight policies above remain the **validated core** and the default return
of `get_policies()`, in their original order. This is deliberate on three
counts: `build_news_schedule` round-robins in list order, so appending would
change policy→week mapping for every existing C2/C3 run; M&P 2021 can only
validate the pre-2026 landscape; and SG Child Support Package *supersedes* two
core instruments (`policy.SUPERSEDED_BY`), so mixing eras double-counts the
same support. Opt in with `get_policies(include_ndr2026=True)` or
`era="ndr2026"`, and write NDR corpus articles to a **separate** file
(`generate_news_corpus.py --era ndr2026 --output ...`) — a policy with no
corpus article silently falls back to legacy description text, which is not a
comparable stimulus. See `docs/sandbox_policy.md`.

---

## Dataset Pipeline Summary

The simulation input `agents_final_100.json` is built in three stages from
`nvidia/Nemotron-Personas-Singapore`. The old single-notebook pipeline is archived
under `history/` and is **superseded** by the validation-driven build below.

### Stage A — Source & filtering
Load the Nemotron-Personas-Singapore dataset (HuggingFace) and keep rows with
`age` 21–45, `gender` ∈ {Male, Female}, and raw marital ∈ {Single, Married}. This
filtered pool feeds both the (historical) production agents and the validation sample.

### Stage B — Validation sample + 5-rater labelling (`src/validation/`)
1. **`sample_personas.py`** — stratified draw of **500** personas matching the M&P
   2021 gender × marital proportions (the agent targets ×2.5): Male/Single 125,
   Female/Single 118, Male/Married 115, Female/Married 142. Seeded (`RANDOM_SEED=42`),
   shuffled, labelled `V001`–`V500`, `uuid` kept for traceability →
   `outputs/validation/validation_personas_500.csv`. (Seed-nested, so it is a
   superset of the earlier 200 production agents.)
2. **`run_inference.py`** — each of **5 rater models** (`nemotron-120b`,
   `gpt-4o-mini`, `claude-haiku-4.5`, local `llama-3.1-8b`, local `qwen2.5-14b`)
   infers, per persona, a `financial_security_score` (1–5) + reasoning and — for
   singles — a `relationship_status` (Single/Dating) + reasoning (temperature 0.2,
   prompt set **v2**) → `preds_<model>_v2.jsonl`.
3. **Consensus + agreement** (`validation_analysis.ipynb`; `pilot_report.py` does the
   v1-vs-v2 comparison) aggregate the 5 raters into `consensus_labels.csv`
   (`fin_consensus_median`, `fin_spread` = max − min, `n_raters_fin`, plus relationship
   consensus fields). **Findings:** the financial score is reliable (Krippendorff
   α ≈ 0.738, ordinal); Single vs Dating could **not** be recovered from persona text
   (low agreement) — hence it is randomised downstream.

### Stage C — Final 100-agent build (`src/build_final_agents.py`)
- **Eligible pool:** personas with **unanimous** financial consensus
  (`fin_spread == 0`; ≈190 of the 500); their scores intrinsically span only **2–4**.
- **Stratified draw of 100** on gender × marital to the scaled M&P targets:
  Male/Single 25, Female/Single 24, Male/Married 23, Female/Married 28
  (→ 49 Single / 51 Married, Male 48 / Female 52). Gender × marital is the **only**
  enforced dimension; age and education are reported as M&P reference deltas, **not fitted**.
- **Relationship status:** Married passes through; the 49 singles are split into
  **24 Dating / 25 Single by seeded random assignment** (the panel could not recover Dating).
- `financial_security_score` = the unanimous consensus; `financial_security_reasoning`
  = gpt-4o-mini's text; beliefs initialise neutral (`attitude=norm=pbc=3`,
  `fertility_intention_dist=None`); provenance fields attached.
- Output → `agents_final_100.json` (`agent_001`…`agent_100`, sorted by persona id).

**Random seed:** `RANDOM_STATE = 42` drives both the stratified draw and the
Single/Dating split in a fixed, documented call order — load-bearing for
reproducibility. Do not change it unless re-running the pipeline intentionally.

---

## LLM Prompting Conventions

### Seed Memory Generation (Step 1 of belief initialisation)

Generate **5 profile seed memories** per agent in first-person natural language, derived from their static profile. These represent stable background beliefs.

Example for a 30-year-old, dating, career-focused agent with moderate income waiting for BTO:
```
"I am currently dating and may consider marriage in the future."
"I am career-focused and worried about promotion timing."
"I have a moderate income but housing is not fully settled."
"My parents expect me to have children eventually."
"I may get flexible work arrangements after my promotion."
```

### TPB Relevance Annotation

Prompt the LLM to score each memory's relevance to each TPB construct on [0, 1]:

```
Attitude prompt: Does this memory describe positive or negative expected outcomes,
  emotions, trade-offs, benefits, or costs of having a child within the next 3 years?

Subjective Norm prompt: Does this memory describe what important others think the agent
  should do, or what important others are doing regarding marriage, childbirth,
  or delaying children?

PBC prompt: Does this memory describe resources, barriers, constraints, or support
  that make having a child feel easier or harder within the next 3 years?
```

Output format:
```json
{"attitude_relevance": 0.2, "norm_relevance": 0.1, "pbc_relevance": 0.9}
```

These judge prompts (`CONSTRUCT_PROMPTS`, mirroring the construct definitions) are used
verbatim by the `llm` scorer and by the `hybrid` reranker — **do not broaden them**.
The `cosine` / `hybrid`-prefilter stage instead embeds a separate, recall-broadened
set of anchors (`CONSTRUCT_EMBED_PROMPTS`) that enumerate the concrete ways each
construct shows up in memories — `subjective_norm` especially (parental/peer/referent
expectations, injunctive vs descriptive, social messaging) — because bare-definition
cosine underranks relational norm content. Broadening the embed anchor is safe: in
`hybrid` mode cosine only decides *which* memories the LLM judges (recall); the LLM
still supplies the per-construct scores (precision).

### Belief State Update (each timestep) — two decoupled LLM calls

TPB scores and the fertility-intention distribution are generated by **two
separate LLM calls** so the TPB→intention relationship can be *measured* rather
than *instructed* (the mediation design). Both calls see the same context
(retrieved memories + this week's new messages); only the previous state shown to
each differs.

**Call 7a — TPB scores** (`build_tpb_update_prompt` → `TPB_UPDATE_SYSTEM`)
- Input: agent profile, previous **TPB scores only**, retrieved memories (≤20), new messages.
- Output: `attitude_score`, `subjective_norm_score`, `pbc_score` (each 1–5), `reflection_memory` (one first-person sentence).

**Call 7b — Fertility intention** (`build_intention_update_prompt` → `INTENTION_UPDATE_SYSTEM`)
- Input: agent profile, previous **intention distribution only**, retrieved memories (≤20), new messages.
- Output: `fertility_intention` (5-float distribution summing to 1.0).
- **Must not** reference `attitude`/`subjective_norm`/`pbc` or instruct consistency with them — that isolation is what makes the TPB→intention link an earned measurement.

---

## Validation Checks

Before interpreting simulation results, verify:

1. **External validity (primary analysis)**: does the simulated per-category ranking reproduce real-world evidence? See `outputs/analysis/real_world_validation_table.md` for the policy → closest-finding → direction-of-agreement map. Three tiers, in descending strength:
   - **Experimental** — Wang & Dong (2024), *Eur. J. Population* 40(1):33: survey experiment, N = 1,092 Singaporeans aged 25–39, OR 1.55–1.79 for flexible work, work-family-conflict mechanism. The closest thing to a real benchmark in the policy set.
   - **Behavioural** — Yeung et al. (2023), *J. Marriage and Family*, N = 1,835: paternity-leave takers were *no* more likely to have another child. A null that conflicts with the perceived-conducive ranking, so within-category evidence is heterogeneous.
   - **Stated preference** — M&P Survey 2021: financial cost is the most-cited barrier; 81–90% said flexible work eased family formation, 77% said the same of paternity leave.

   **Vintage caveat, must be stated in any write-up:** all eight instruments took effect Dec 2024 – Apr 2026, *after* the 2021 survey. The validation therefore operates at the level of **instrument type**, not specific scheme — the survey asked about flexible work and paternity leave directly, and the modelled instruments are enhancements to those same types. Compare category-level rankings, not scheme-level effect sizes.

   Three of eight policies (Large Family Scheme, Child LifeSG Credits, Infant Childminding Pilot) have **no** real-world evaluation. Flag as untested rather than forcing a match.
2. **Mechanism validity (secondary)**: where TPB runs are reported, use the per-policy `expected_pathways` in `src/sandbox/policy.py` (analysis-only metadata, never shown to agents):
   - **Specificity** — does a policy raise its hypothesised construct *more* than the others? Verified: 43/45 articles move `pbc` hardest, and this survives construct-blind retrieval, so it is content-driven rather than a retrieval artifact.
   - **Mediation** — **tested and not supported under policy input** (see Research Question). Real and ablation-robust under social-only.
   - **Sign / rank** — construct→intention slopes should be positive. The human benchmark is contested: Ajzen & Klobas (2013) report per-country coefficients with no stable ordering (norm outranks attitude in Germany), while a 2025 meta-analysis (Liang et al., *Advances in Psychological Science* 33(11):1926–1941; 33 studies, N = 43,427) reports attitude .41 > norm .30 > PBC .23. Cite the specific source rather than a shorthand ordering.

   Controlled unit tests remain useful as quick checks —
   - Career-delay memory → `pbc_score` should decrease.
   - Parental-pressure memory → `subjective_norm_score` should increase.
   - Financial-support policy memory → `pbc_score` should increase.
3. **Annotation validity**: `financial_security_score` is validated (5 raters, Krippendorff α ≈ 0.738). The equivalent human-coded benchmark for TPB relevance labels has **not** been built and remains an open limitation.
4. **Internal validity**: C0 must be flat with zero input (update gating); effects are measured net of that control.

---

## Key Constraints & Scope

**In scope:**
- Single social network (one layer)
- 100 agents, 1-week timesteps
- Four experimental conditions (C0–C3)
- Two agent architectures: TPB belief layer (`engines/engine.py`) and no-TPB VacSim-direct (`engines/engine_direct.py`), compared as a robustness dimension
- Policy category as the primary treatment (financial / caregiving)
- Fertility *intention* only (not actual birth behaviour)

**Out of scope (do not model):**
- Actual childbirth or fertility outcomes
- Marriage formation or partner matching
- Multiple overlapping networks (family, workplace, online)
- Immigration or non-resident population dynamics
- Predicting Singapore's real TFR

## File Conventions

| File | Purpose |
|---|---|
| `fark_agent.ipynb` | Data pipeline: Nemotron → 100 calibrated agents |
| `agents_initialised.json` | v1-pipeline agent profiles (kept for provenance; superseded as sim input) |
| `agents_final_100.json` | Final 100-agent pool from the validated v2 consensus (see note below); the pristine profiles (neutral beliefs, no seeds) |
| `agents_final_100_seeded.json` | **Current simulation input for runs.** `agents_final_100.json` + seed memories + the frozen t=0 baseline, produced once by `driver.py --init-only`; loaded by every condition (C0–C3) so they start identical |
| `CLAUDE.md` | This file — project reference for AI agents and developers |
| `src/build_final_agents.py` | Builds `agents_final_100.json` from validation outputs (reproducible, seed 42) |
| `src/driver.py` | Simulation runner CLI (conditions C0–C3) |
| `src/generate_social_network.py` | One-off LLM-generated directed friendship network (VacSim style) |
| `src/LLM_judge.py` | TPB relevance scorer, switchable `llm` / `cosine` / `hybrid` (cosine prefilter top-K union → LLM rerank, cached). Cosine uses the recall-broadened `CONSTRUCT_EMBED_PROMPTS`; the LLM judge uses the faithful `CONSTRUCT_PROMPTS` |
| `src/engines/engine.py` | Weekly simulation loop (perceive → retrieve → update → post → save) |
| `src/sandbox/agent.py` | 3-layer fertility agent (static profile, TPB beliefs, memory stream) |
| `src/sandbox/lesson.py` | Memory object (CLAUDE.md schema) + saliency decay + retrieval formula |
| `src/sandbox/news.py` | Policy news items + weekly news schedule |
| `src/sandbox/policy.py` | Singapore policy instruments with expected TPB pathways |
| `src/sandbox/prompts.py` | All LLM prompt templates (seed memories, TPB relevance, decoupled TPB + intention updates, posts) |
| `src/sandbox/tweet.py` | Fertility-related social post (visible to followers at t+1) |
| `src/utils/generate_utils.py` | Configurable LLM client (.env: OpenRouter or local endpoint) |
| `src/utils/network_utils.py` | Social network JSON save/load |
| `outputs/` | Generated artifacts, grouped by type: `runs/` (simulation `<run_name>.json` + `.log`, with throwaway smoke runs under `runs/smoke/`), `networks/` (`social_network_*.json` + `.log`), `analysis/` (comparison/inspection reports + CSV/PNG), `validation/` (rater preds/logs + consensus) |

**Final agent pool (`agents_final_100.json`).** After the multi-LLM validation
study, the simulation input is built by `src/build_final_agents.py` (not the
notebooks). It takes the 190 validation personas whose `financial_security_score`
was **unanimous across all 5 raters** (`fin_spread == 0`), samples **100**
stratified on gender × marital to the scaled M&P 2021 targets (49 Single / 51
Married; M 48 / F 52), and assigns relationship status by seed (`RANDOM_STATE=42`,
24/25 Single↔Dating among singles) because the LLM panel could not recover Dating
from persona text. `financial_security_score` = the unanimous consensus value
(range is intrinsically **2–4**; the panel never near-unanimously assigns 1 or 5);
`financial_security_reasoning` = gpt-4o-mini's text. Each agent additionally
carries provenance fields ignored by `sandbox.agent.Agent`: `source_persona_id`,
`uuid`, `fin_consensus_n_raters`, `fin_consensus_spread`. The `agents_final_100`
social network has been regenerated and validated: C1/C3 use
`outputs/networks/social_network_qwen_noarea.json` (demographics-only,
planning-area field ablated — see
`outputs/analysis/network_validation/network_area_decision.md`).
---

## Session Handoff (any device — read this first in a new session)

The project moves between a Windows laptop, a Mac, and a GPU server; the repo
folder syncs via OneDrive. An AI agent starting fresh on ANY device should
read, in order:

1. **`docs/agent_memory/MEMORY.md`** — index of the working-memory notes
   (decisions, gotchas, live status), exported from the primary agent's
   memory on 2026-07-15. `c2-tpb-ratchet-drift.md` is the live status note.
2. **`outputs/README.md`** — map of the outputs tree; canonical artifacts
   marked ►.
3. **`outputs/analysis/progress_summary_network_to_news.md`** — chronological
   record of every validation, result, and decision (9–15 Jul 2026), with a
   TL;DR and the current run queue.

Platform notes:
- The `.venv/` in this folder is **Windows-only**. Do not delete it (OneDrive
  propagates deletions); on Mac create a venv outside OneDrive
  (`python3 -m venv ~/.venvs/research`).
- LLM runs execute on the GPU server (school vLLM, Qwen2.5-14B, ports
  8101–8104 via `serve_qwen.sh`); the server is password-SSH only, so agents
  hand one-line commands to the user. Files move laptop↔server via OneDrive
  sync and/or git (note: `outputs/` is gitignored except force-added
  artifacts).
- The VacSim reference clone is per-device (Windows: `C:\Users\chong\VacSim`;
  elsewhere clone github.com/abehou/VacSim to `~/VacSim`, outside OneDrive).
- Commits are authored by the user only — never add AI co-author lines.

## References (Short Form)
- Singapore TFR 2025: 0.87 (SingStat)
- M&P Survey 2021: Key Findings, Population Singapore
- Theory of Planned Behaviour: Ajzen (1991)
- TPB applied to fertility intentions: Ajzen & Klobas (2013), *Demographic Research* 29:203–232
- Social contagion in fertility: Balbo & Barban (2012)
- VacSim simulation framework: memory saliency and social post mechanics
- Nemotron-Personas-Singapore: nvidia/Nemotron-Personas-Singapore (HuggingFace)
- **Flexible work → fertility intention (experimental):** Wang & Dong (2024), *European Journal of Population* 40(1):33 — Singapore survey experiment, N = 1,092, OR 1.55–1.79
- **Paternity leave → behaviour (null):** Yeung et al. (2023), *Journal of Marriage and Family* — Singapore longitudinal, N = 1,835
- **TPB construct ordering (meta-analysis):** Liang, Zhao, Zhao, Yue & He (2025), *Advances in Psychological Science* 33(11):1926–1941 — 33 studies, N = 43,427; attitude .41 > norm .30 > PBC .23
- NDR 2026 marriage & parenthood measures: Population Singapore (17 Aug 2026)

# VacSim Mapping

VacSim Concept → This Project

Agent
→ Fertility Agent

Tweet
→ Fertility-related Post

News
→ Policy News

Belief Score
→ TPB Scores

Memory Saliency
→ importance × decay

Reflection
→ fertility reflection memory

Social Network
→ fixed directed exposure graph

# Agent Operating Rules
When making implementation decisions:
1. Follow this document.
2. Preserve reproducibility.
3. Prefer simple solutions over complex solutions.
4. Do not introduce new agent attributes without updating schemas.
5. Do not change TPB definitions.
6. Do not change memory retrieval formula.
7. Ask before changing experimental conditions.
8. Keep all outputs traceable.