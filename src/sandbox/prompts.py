"""All LLM prompt templates for the fertility ABM, following the
LLM Prompting Conventions in CLAUDE.md.

Every build_* function returns (system_prompt, user_prompt) ready for
LLMClient.chat_json().
"""

from utils.utils import compile_enumerate

# ──────────────────────────────────────────────────────────────────────────
# Agent profile rendering
# ──────────────────────────────────────────────────────────────────────────

def profile_to_str(agent):
    """Render an agent's static profile (Layer 1) for prompt context."""
    return (
        f"Age: {agent.age}\n"
        f"Gender: {agent.gender}\n"
        f"Relationship status: {agent.relationship_status}\n"
        f"Education: {agent.education}\n"
        f"Occupation: {agent.occupation}\n"
        f"Industry: {agent.industry}\n"
        f"Planning area (residence): {agent.planning_area}\n"
        f"Financial security (1=Low, 5=Upper): {agent.financial_security_score}\n"
        f"Persona: {agent.general_persona}\n"
        f"Cultural background: {agent.cultural_background}\n"
        f"Hobbies and interests: {agent.hobbies_and_interests}\n"
        f"Career goals and ambitions: {agent.career_goals}"
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. Seed memory generation (belief initialisation, step 1)
# ──────────────────────────────────────────────────────────────────────────

SEED_MEMORY_SYSTEM = """You are simulating the inner life of a Singapore resident.
Given their profile, write exactly 5 short first-person memories that represent
stable background beliefs relevant to marriage, family, career, finances, housing,
and having children. These are the person's own thoughts, in natural everyday
language (one sentence each). Stay consistent with the profile; do not invent
major facts (children, divorce, illness) not implied by it.

For each memory also estimate its importance to the person's life on a 0-1 scale
(0 = barely memorable, 1 = life-changing).

Respond with JSON only:
{"memories": [{"memory_text": "...", "importance": 0.7}, ... 5 items ...]}"""


def build_seed_memory_prompt(agent):
    """(system, user) prompt asking the LLM for 5 first-person seed memories."""
    user = (
        "Profile of the person:\n"
        f"{profile_to_str(agent)}\n\n"
        "Write the 5 first-person seed memories as JSON."
    )
    return SEED_MEMORY_SYSTEM, user


# ──────────────────────────────────────────────────────────────────────────
# 2. TPB relevance annotation (verbatim construct prompts from CLAUDE.md)
# ──────────────────────────────────────────────────────────────────────────

ATTITUDE_PROMPT = (
    "Does this memory describe positive or negative expected outcomes, emotions, "
    "trade-offs, benefits, or costs of having a child within the next 3 years?"
)

NORM_PROMPT = (
    "Does this memory describe what important others think the agent should do, "
    "or what important others are doing regarding marriage, childbirth, or "
    "delaying children?"
)

PBC_PROMPT = (
    "Does this memory describe resources, barriers, constraints, or support that "
    "make having a child feel easier or harder within the next 3 years?"
)

TPB_RELEVANCE_SYSTEM = f"""You are an annotator for a study based on the Theory of
Planned Behaviour (TPB). Score how relevant a memory is to each TPB construct on
a continuous [0, 1] scale (0 = not relevant at all, 1 = highly relevant).

Attitude: {ATTITUDE_PROMPT}
Subjective Norm: {NORM_PROMPT}
Perceived Behavioural Control (PBC): {PBC_PROMPT}

Respond with JSON only:
{{"attitude_relevance": 0.2, "norm_relevance": 0.1, "pbc_relevance": 0.9}}"""


def build_relevance_prompt(memory_text):
    """(system, user) prompt asking the LLM to score one memory's TPB relevance."""
    user = f'Memory: "{memory_text}"\n\nScore its TPB relevance as JSON.'
    return TPB_RELEVANCE_SYSTEM, user


# Construct prompts reused by the LLM-as-judge relevance scorer (they mirror the
# canonical TPB construct definitions in CLAUDE.md — do not broaden these).
CONSTRUCT_PROMPTS = {
    "attitude": ATTITUDE_PROMPT,
    "norm": NORM_PROMPT,
    "pbc": PBC_PROMPT,
}

# Recall-broadened anchors used ONLY by the cosine / hybrid-prefilter scorer
# (RelevanceScorer._score_cosine). Decoupled from CONSTRUCT_PROMPTS on purpose: the
# cosine stage needs recall, not the exact definition, so these enumerate the many
# concrete ways each construct surfaces in first-person memories — pulling concretely
# phrased memories closer to the anchor in embedding space. `subjective_norm` is
# especially enriched because it is relational and cosine underranks it. Broadening
# here is safe: in hybrid mode cosine only shortlists candidates; the LLM judge
# (which still uses the faithful CONSTRUCT_PROMPTS) supplies the final precision.
CONSTRUCT_EMBED_PROMPTS = {
    "attitude": (
        "Personal feelings and evaluations about having a child in the next few "
        "years: whether it would be fulfilling, joyful, and meaningful or a burden "
        "and a sacrifice; the emotional rewards and the costs; trade-offs with "
        "career, freedom, lifestyle, finances, and personal goals; hopes, worries, "
        "excitement, reluctance, or ambivalence about becoming a parent; whether "
        "raising a child feels worth it."
    ),
    "norm": (
        "Social expectations and pressure about marriage and having children coming "
        "from important people in one's life: parents and in-laws wanting "
        "grandchildren or asking when you will have kids; a spouse's or partner's "
        "wishes; what friends, siblings, relatives, colleagues, and peers expect or "
        "think you should do; friends, siblings, and peers who are themselves getting "
        "married, pregnant, or having babies; feeling that having children is normal, "
        "expected, or the done thing in one's family, community, or generation; "
        "religious, cultural, or government messaging that encourages marriage and "
        "parenthood; comparing oneself to other people's family choices."
    ),
    "pbc": (
        "Sense of being able to afford, manage, and cope with having and raising a "
        "child: money, income, savings, cost of living; housing and getting a flat; "
        "childcare availability and cost; job security, career stability, work-life "
        "balance, flexible work arrangements, and parental leave; time, energy, "
        "health, and support from family, a partner, or a helper; practical barriers "
        "or supports that make having a child feel easier or harder to handle."
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# 3. Perception: turn incoming news / social posts into memories
# ──────────────────────────────────────────────────────────────────────────

PERCEPTION_SYSTEM = """You are simulating how a Singapore resident internalises
something they just read. Given their profile and an incoming message, write ONE
short first-person memory (one sentence) capturing what this person personally
takes away from the message, filtered through their own circumstances. Also
estimate the importance of this takeaway to them on a 0-1 scale
(0 = barely memorable, 1 = life-changing).

Respond with JSON only:
{"memory_text": "...", "importance": 0.5}"""


def build_perception_prompt(agent, message_text, message_kind):
    """(system, user) prompt turning an incoming message into a 1-sentence memory.

    `message_kind` is a human label for the source ("news report" / "social
    media post") so the prompt reads naturally.
    """
    user = (
        "Profile of the person:\n"
        f"{profile_to_str(agent)}\n\n"
        f"They just read this {message_kind}:\n\"{message_text}\"\n\n"
        "Write their one-sentence first-person takeaway memory as JSON."
    )
    return PERCEPTION_SYSTEM, user


# ──────────────────────────────────────────────────────────────────────────
# 4. Belief state update (each timestep) — DECOUPLED into two independent calls.
#    TPB scores and the fertility-intention distribution are generated by two
#    SEPARATE LLM calls so the TPB->intention link can be *measured* rather than
#    *instructed* (mediation design). Both calls see the same context (retrieved
#    memories + this week's messages), but the intention call never sees the
#    numeric attitude/norm/pbc scores, and the TPB call never sees intention.
# ──────────────────────────────────────────────────────────────────────────

TPB_UPDATE_SYSTEM = """You are simulating the fertility-related beliefs of a
Singapore resident using the Theory of Planned Behaviour (TPB). Based on their
profile, current TPB scores, and retrieved memories, output their updated TPB
scores.

Definitions (all scores on a 1-5 scale):
- attitude_score: how positively or negatively they evaluate having a child
  (1 = very negative, 3 = neutral, 5 = very positive).
- subjective_norm_score: how much social pressure / perceived normalcy they feel
  from important referents about having children (1 = none, 5 = very strong).
- pbc_score: how capable they feel of having and raising a child given their
  resources and constraints (1 = not capable at all, 5 = fully capable).

Update scores gradually and realistically: a single week of input should shift
scores by small amounts unless the memories are truly life-changing. Stay
consistent with the person's profile and circumstances.

Also write reflection_memory: ONE first-person sentence summarising how this
week's experiences changed (or reinforced) their thinking about having children.

Respond with JSON only:
{"attitude_score": 3.2, "subjective_norm_score": 3.0, "pbc_score": 2.8,
 "reflection_memory": "..."}"""


INTENTION_UPDATE_SYSTEM = """You are simulating the fertility intention of a
Singapore resident. Based on their profile, their previous fertility intention,
and the experiences (memories and messages) they had this week, output their
updated fertility intention.

fertility_intention is a probability distribution over 5 ordinal intention levels
[p1, p2, p3, p4, p5] that sums to 1.0, where
1 = no child intention, 2 = weak/unlikely, 3 = uncertain, 4 = likely,
5 = strong intention.

Judge their intention holistically from their overall situation and what they
experienced this week. Update gradually and realistically: a single week should
shift the distribution only a little unless the experiences are truly
life-changing. Briefly explain your reasoning in one sentence.

Respond with JSON only:
{"reasoning": "...", "fertility_intention": [0.1, 0.2, 0.4, 0.2, 0.1]}"""


def _belief_context_lines(retrieved_lessons, new_messages):
    """Shared context block (retrieved memories + this week's messages).

    Both decoupled belief calls (TPB and intention) see exactly the same
    experiences this week; only the *previous state* shown to each differs.
    """
    memory_lines = [
        f"(t={l.created_timestep}, {l.source_type}) {l.memory_text}"
        for l in retrieved_lessons
    ]
    parts = [
        compile_enumerate(memory_lines, header="Retrieved memories (most relevant first)")
        if memory_lines else "Retrieved memories: none",
    ]
    if new_messages:
        parts += ["", compile_enumerate(new_messages, header="New messages read this week")]
    return parts


def build_tpb_update_prompt(agent, retrieved_lessons, new_messages, current_timestep):
    """(system, user) prompt for the weekly TPB-score update (no intention).

    Shows the current attitude/norm/pbc scores (NOT the intention distribution),
    the retrieved memories, and any new messages. The LLM returns the updated
    three scores + a reflection sentence.
    """
    belief = agent.belief_state
    parts = [
        "Profile of the person:",
        profile_to_str(agent),
        "",
        f"Current week of the simulation: {current_timestep}",
        "",
        "Current TPB scores:",
        f"- attitude_score: {belief['attitude_score']}",
        f"- subjective_norm_score: {belief['subjective_norm_score']}",
        f"- pbc_score: {belief['pbc_score']}",
        "",
        *_belief_context_lines(retrieved_lessons, new_messages),
        "",
        "Output the updated TPB scores and reflection_memory as JSON.",
    ]
    return TPB_UPDATE_SYSTEM, "\n".join(parts)


def build_intention_update_prompt(agent, retrieved_lessons, new_messages, current_timestep):
    """(system, user) prompt for the weekly fertility-intention update.

    Deliberately shows NONE of the numeric TPB scores — only the person's
    previous intention distribution, the retrieved memories, and any new
    messages — so the TPB->intention link is measured, not instructed.
    """
    belief = agent.belief_state
    prev = belief["fertility_intention_dist"]
    prev_line = (
        f"Previous fertility intention distribution: {prev}"
        if prev is not None
        else "This is the first assessment of their fertility intention."
    )
    parts = [
        "Profile of the person:",
        profile_to_str(agent),
        "",
        f"Current week of the simulation: {current_timestep}",
        "",
        prev_line,
        "",
        *_belief_context_lines(retrieved_lessons, new_messages),
        "",
        "Output the updated fertility_intention distribution as JSON.",
    ]
    return INTENTION_UPDATE_SYSTEM, "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# 4b. Baseline belief initialisation (t=0) — DECOUPLED like the weekly update.
#     Run ONCE, before any week, over the seed memories only. Unlike the weekly
#     update these prompts *establish* the starting scores from scratch: there is
#     no "current 3/3/3" anchor and no "shift gradually" instruction, so the LLM
#     is free to place each agent across the full 1-5 range based on their profile
#     and seeds. They produce a grounded, differentiated baseline to measure
#     policy/social effects *from*. Scores only — no reflection memory is written,
#     so the t=0 memory stream stays the pure seed memories.
# ──────────────────────────────────────────────────────────────────────────

BASELINE_TPB_SYSTEM = """You are establishing the INITIAL fertility-related TPB
beliefs of a Singapore resident, using the Theory of Planned Behaviour (TPB). Based
on their profile and their stable background (seed) memories, output their starting
TPB scores.

Definitions (all scores on a 1-5 scale; 3 is the neutral midpoint of the scale):
- attitude_score: how positively or negatively they evaluate having a child
  (1 = very negative, 3 = neutral, 5 = very positive).
- subjective_norm_score: how much social pressure / perceived normalcy they feel
  from important referents about having children (1 = none, 5 = very strong).
- pbc_score: how capable they feel of having and raising a child given their
  resources and constraints (1 = not capable at all, 5 = fully capable).

Set each score to reflect what THIS person's profile and background beliefs
genuinely imply. Use the FULL 1-5 range where warranted — do NOT default everyone to
the middle. Two people in clearly different circumstances should receive clearly
different scores.

Respond with JSON only:
{"attitude_score": 2.4, "subjective_norm_score": 3.6, "pbc_score": 2.0}"""


BASELINE_INTENTION_SYSTEM = """You are establishing the INITIAL fertility intention
of a Singapore resident. Based on their profile and their stable background (seed)
memories, output their starting fertility intention.

fertility_intention is a probability distribution over 5 ordinal intention levels
[p1, p2, p3, p4, p5] that sums to 1.0, where
1 = no child intention, 2 = weak/unlikely, 3 = uncertain, 4 = likely,
5 = strong intention.

Judge their intention holistically from their overall situation. Use the FULL range
where warranted — do NOT default to a flat or uniformly uncertain distribution
unless that genuinely fits this person. Briefly explain your reasoning in one
sentence.

Respond with JSON only:
{"reasoning": "...", "fertility_intention": [0.3, 0.3, 0.2, 0.1, 0.1]}"""


def build_baseline_tpb_prompt(agent, retrieved_lessons):
    """(system, user) prompt for the t=0 baseline TPB scores (no intention).

    Shows the profile + seed memories but NO current-score anchor and NO new
    messages — the baseline is established from the seeds alone.
    """
    parts = [
        "Profile of the person:",
        profile_to_str(agent),
        "",
        "This is the BASELINE assessment, before the simulation begins "
        "(no policy or social input yet).",
        "",
        *_belief_context_lines(retrieved_lessons, []),
        "",
        "Output the baseline TPB scores as JSON.",
    ]
    return BASELINE_TPB_SYSTEM, "\n".join(parts)


def build_baseline_intention_prompt(agent, retrieved_lessons):
    """(system, user) prompt for the t=0 baseline fertility-intention distribution.

    Like the baseline TPB prompt: profile + seed memories, no previous-intention
    anchor, no new messages, no numeric TPB scores (mediation isolation holds at
    baseline too).
    """
    parts = [
        "Profile of the person:",
        profile_to_str(agent),
        "",
        "This is the BASELINE assessment, before the simulation begins "
        "(no policy or social input yet).",
        "",
        *_belief_context_lines(retrieved_lessons, []),
        "",
        "Output the baseline fertility_intention distribution as JSON.",
    ]
    return BASELINE_INTENTION_SYSTEM, "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# 5. Social post (tweet) generation
# ──────────────────────────────────────────────────────────────────────────

TWEET_SYSTEM = """You are simulating a Singapore resident deciding whether to post
on social media this week. Based on their profile, current beliefs, and recent
reflection, decide if they would share something related to marriage, family,
parenting, children, or family-related policies. People do not post every week;
only post if this week genuinely gave them something to say.

If they post, write it in their authentic voice (casual Singapore social-media
tone, 1-3 sentences, no hashtag spam). Do not mention TPB, scores, or the
simulation.

Respond with JSON only:
{"post": true, "text": "..."}  or  {"post": false, "text": null}"""


def build_tweet_prompt(agent, reflection_text, current_timestep):
    """(system, user) prompt asking the LLM whether (and what) the agent posts."""
    belief = agent.belief_state
    user = (
        "Profile of the person:\n"
        f"{profile_to_str(agent)}\n\n"
        f"Current week: {current_timestep}\n"
        f"Current beliefs about having children — attitude: {belief['attitude_score']}/5, "
        f"felt social pressure: {belief['subjective_norm_score']}/5, "
        f"sense of capability: {belief['pbc_score']}/5.\n"
        f"This week's reflection: \"{reflection_text}\"\n\n"
        "Decide whether they post and output JSON."
    )
    return TWEET_SYSTEM, user
