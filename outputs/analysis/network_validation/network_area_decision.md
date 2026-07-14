# Decision: use the no-area network (`social_network_qwen_noarea.json`) for C1/C3

**Date:** 2026-07-11 · **Status:** recommended, pending final sign-off
**Evidence:** `network_validation_social_network_qwen.md` (single-network validation),
`network_comparison_with_area_vs_no_area_vs_with_area_rep2.md` (ablation), both in
this folder. Scripts: `src/validation/validate_network.py`, `compare_networks.py`,
generation flag `generate_social_network.py --no-area`.

## The problem

The social network for C1/C3 is LLM-generated (VacSim style): each agent sees a
one-line demographic profile of the 99 others and names its friends. Structural
validation of the original network (`social_network_qwen.json`) found plausible
degree/connectivity/reciprocity/clustering and significant homophily on age,
gender, relationship status, and education — but **implausibly strong geographic
homophily**: 31.3% of ties join agents in the same planning area, vs 5.0% under a
node-label permutation null (z ≈ +32). Every other homophily dimension sits at
z ≈ 3–5; geography alone is an order of magnitude stronger.

Substantively this is wrong for Singapore: the country is ~50 km across, commutes
are short, and friendships form through school, work, and NS cohorts far more than
through residential proximity. Nothing in the fertility literature predicts
same-planning-area peers dominate fertility-relevant social exposure.

## The ablation

Hypothesis: the LLM was not modelling geography — it was pattern-matching on a
visible `Area:` field ("friends = neighbours") because the profile line offers few
other easy matching cues.

Design: regenerate the network twice (same model Qwen2.5-14B-Instruct, same
prompt, temperature 0.7):

1. **no_area** — identical prompt with the `Area` field removed from every profile
   and from the prompt's schema description.
2. **with_area_rep2** — an exact replicate of the original condition, as the
   noise floor (generation is stochastic at temperature 0.7, so we must know how
   much two same-condition networks differ before interpreting the ablation).

## Results

| | with_area | with_area_rep2 | no_area |
|---|---|---|---|
| same-planning-area tie share | **31.3% (z +32.2)** | 30.1% (z +31.0) | **6.2% (z +1.6)** |
| age \|gap\| years (− = assortative) | 6.51 (z −3.9) | 6.60 (z −3.3) | 6.25 (z −5.1) |
| same-gender share | 0.562 (z +3.6) | 0.557 (z +3.5) | 0.537 (z +2.6) |
| same-relationship share | 0.465 (z +4.6) | 0.463 (z +4.4) | 0.474 (z +5.5) |
| same-education share | 0.504 (z +5.3) | 0.530 (z +6.2) | **0.572 (z +8.8)** |
| directed edges | 824 | 851 | 1015 |
| reciprocity | 0.26 | 0.27 | 0.28 |
| mean clustering | 0.28 | 0.28 | 0.32 |
| weak components | 1 | 1 | 1 |
| zero-in-degree agents | 1 (agent_005) | 1 | 1 (agent_016) |

## Why the no-area network should be used

1. **The geo anchoring is an artifact, demonstrably.** Hiding the field collapses
   same-area ties from 31% to 6.2% — statistically indistinguishable from chance.
   The LLM had no geographic model; it had a string-matching habit.
2. **The difference is not sampling noise.** The two with-area replicates agree
   almost exactly on every metric (area share 31.3% vs 30.1%); the ablation effect
   is ~25 percentage points against a replicate gap of ~1.
3. **Attention reallocates to meaningful cues.** Without `Area`, homophily
   *strengthens* on the dimensions that plausibly matter for fertility-relevant
   peer influence — age (z −3.9 → −5.1), relationship status (+4.6 → +5.5), and
   especially education (+5.3 → +8.8). These are the life-stage cues through which
   social contagion in fertility is theorised to travel (Balbo & Barban 2014).
4. **Structure is preserved or improved.** Still one weak component, reciprocity
   and clustering stable (0.28 / 0.32), no new isolates. The network is somewhat
   denser (mean out-degree 10.2 vs 8.2) — more social exposure per agent, still
   well inside the context budget for belief prompts (retrieval is capped at 20
   memories regardless of in-neighbourhood size).
5. **The alternative is worse for the research question.** C1/C3 measure whether
   peer exposure moves TPB constructs. If 31% of exposure edges are keyed to an
   arbitrary residence label, peer-influence estimates are structured by an
   artifact with no theoretical relevance. Removing it makes the exposure graph a
   cleaner instrument.

## Costs / what we give up

- **Divergence from VacSim's exact profile string** (theirs includes location).
  Mitigation: the change is a single documented field removal, motivated and
  validated by an ablation with a replicate control — a methods strength, not a
  deviation to hide.
- **Zero geographic signal** rather than a realistic weak one. Acceptable: for
  fertility peer influence at n=100 there is no credible way to calibrate a
  "correct" small geo effect, and chance-level is closer to truth than 6× chance.
- **A different zero-in-degree agent** (agent_016 instead of agent_005; posts
  never read, one agent out of 100). Generation quirk present in every variant;
  note as limitation, do not hand-patch (would break LLM-generated provenance).

## Appendix: persona-inclusion test (2026-07-11, run 2026-07-13 analysis)

A follow-up test appended each agent's narrative `general_persona` to the
profile line (`--persona --no-area`, ~10k-token prompts, vLLM relaunched at
MAX_MODEL_LEN=16384). Generation was clean (all 100 agents LLM-chosen, no
fallbacks). Comparison (`network_comparison_no_area_vs_persona_noarea_vs_with_area.md`):

| | no_area | persona_noarea |
|---|---|---|
| age \|gap\| years | 6.25 (z −5.1) | 6.54 (z −3.8) |
| same-gender | 0.537 (z +2.6) | **0.621 (z +7.4)** |
| same-relationship | 0.474 (z +5.5) | 0.447 (z +3.9) |
| same-education | 0.572 (z +8.8) | 0.465 (z +3.6) |
| same-planning-area | 0.062 (z +1.6) | 0.058 (z +1.0) |
| reciprocity / clustering | 0.28 / 0.32 | 0.19 / 0.23 |
| zero-in-degree | 1 | 2 |

The persona text **dilutes the life-stage cues** (age, relationship, education
homophily all weaken) and replaces them with a strong **gender** anchor
(z +2.6 → +7.4) — plausibly because Nemotron personas carry gendered
interest/hobby narratives that the LLM matches on.

**Caveat — the gender signal is not proven "bad".** Unlike planning area, the
gender jump has no ablation-with-replicate behind it, and strong gender
homophily is well documented in real adult friendship networks (same-gender
shares of ~0.6+; McPherson, Smith-Lovin & Cook 2001), so 0.621 could even be
*more* realistic than the no-area network's 0.537. What IS established: the
jump is far outside the replicate noise band (the two with-area replicates
differ by only ~0.005 on gender share), so the persona text really drives it —
but whether as legitimate signal or as stereotyped-content string-matching is
untested. The rejection therefore rests on three other grounds:

1. **Life-stage dilution.** Age, relationship-status, and education homophily —
   the channels through which fertility contagion is theorised to travel
   (Balbo & Barban 2014) — all weaken under persona inclusion.
2. **Structural degradation in the unrealistic direction.** Reciprocity
   0.28 → 0.19 and clustering 0.32 → 0.23; real friendship networks are highly
   reciprocal and clustered (McPherson et al. 2001), and a second
   zero-in-degree agent appears.
3. **Opacity.** With demographics-only profiles the exposure graph's
   construction is fully auditable; with free-text personas the matching
   channel cannot be examined.

**Persona inclusion is rejected** as the default (it also diverges from
VacSim's demographics-only profile string). The no-area, demographics-only
network remains the recommendation. If the gender question is revisited, the
decisive tests are: (a) a persona run with the explicit `Gender:` field hidden
(narrative alone carrying gender ⇒ stereotyped-content artifact), (b) a persona
replicate for the noise floor, (c) a qualitative audit of chosen friends
against persona text.

## Reproducibility

- Generation: `python generate_social_network.py --no-area --output
  ../outputs/networks/social_network_qwen_noarea.json` (Qwen2.5-14B-Instruct,
  local vLLM, temperature 0.7, retry-then-random-fallback k=5, seed 42 for the
  fallback RNG only — LLM choices themselves are sampled).
- Validation/comparison: `python validation/compare_networks.py --networks
  <with_area> <no_area> <with_area_rep2> --labels ...` (permutation null n=2000,
  seed 42).
- Wiring (posts flow t−1 → t) is unchanged; only the edge list differs.

## References

- Balbo, N., & Barban, N. (2014). Does fertility behavior spread among friends?
  *American Sociological Review*, 79(3), 412–431.
  https://doi.org/10.1177/0003122414531596
  (Fertility peer effects travel through life-stage-similar network partners —
  the basis for weighting age/relationship/education homophily over geography
  and gendered interests.)
- McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a feather:
  Homophily in social networks. *Annual Review of Sociology*, 27, 415–444.
  https://doi.org/10.1146/annurev.soc.27.1.415
  (Empirical benchmarks for homophily in real friendship networks — strong
  gender homophily is normal, which is why the persona network's gender signal
  is not itself evidence of an artifact; also the expectation that friendship
  networks are highly reciprocal and clustered.)
- Ajzen, I., & Klobas, J. (2013). Fertility intentions: An approach based on
  the theory of planned behavior. *Demographic Research*, 29, 203–232.
  https://doi.org/10.4054/DemRes.2013.29.8
  (Project-wide TPB-for-fertility framing; why the exposure graph exists at
  all — C1/C3 test whether peer exposure moves TPB constructs.)
