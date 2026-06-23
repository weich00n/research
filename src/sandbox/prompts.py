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


# Construct prompts reused by the cosine-similarity relevance scorer.
CONSTRUCT_PROMPTS = {
    "attitude": ATTITUDE_PROMPT,
    "norm": NORM_PROMPT,
    "pbc": PBC_PROMPT,
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
# 4. Belief state update (each timestep)
# ──────────────────────────────────────────────────────────────────────────

BELIEF_UPDATE_SYSTEM = """You are simulating the fertility-related beliefs of a
Singapore resident using the Theory of Planned Behaviour (TPB). Based on their
profile, current TPB scores, and retrieved memories, output their updated state.

Definitions (all scores on a 1-5 scale):
- attitude_score: how positively or negatively they evaluate having a child
  (1 = very negative, 3 = neutral, 5 = very positive).
- subjective_norm_score: how much social pressure / perceived normalcy they feel
  from important referents about having children (1 = none, 5 = very strong).
- pbc_score: how capable they feel of having and raising a child given their
  resources and constraints (1 = not capable at all, 5 = fully capable).
- fertility_intention: a probability distribution over 5 ordinal intention
  levels [p1, p2, p3, p4, p5] that sums to 1.0, where
  1 = no child intention, 2 = weak/unlikely, 3 = uncertain, 4 = likely,
  5 = strong intention.

Update beliefs gradually and realistically: a single week of input should shift
scores by small amounts unless the memories are truly life-changing. Stay
consistent with the person's profile and circumstances.

Also write reflection_memory: ONE first-person sentence summarising how this
week's experiences changed (or reinforced) their thinking about having children.

Respond with JSON only:
{"attitude_score": 3.2, "subjective_norm_score": 3.0, "pbc_score": 2.8,
 "fertility_intention": [0.1, 0.2, 0.4, 0.2, 0.1],
 "reflection_memory": "..."}"""


def build_belief_update_prompt(agent, retrieved_lessons, new_messages, current_timestep):
    """(system, user) prompt for the weekly TPB update.

    Assembles the user message from: the agent profile, the current TPB scores,
    the retrieved memories (rendered as a numbered list via compile_enumerate),
    and any new messages read this week. The LLM returns the new scores +
    intention distribution + a reflection sentence.
    """
    belief = agent.belief_state
    memory_lines = [
        f"(t={l.created_timestep}, {l.source_type}) {l.memory_text}"
        for l in retrieved_lessons
    ]
    parts = [
        "Profile of the person:",
        profile_to_str(agent),
        "",
        f"Current week of the simulation: {current_timestep}",
        "",
        "Current TPB state:",
        f"- attitude_score: {belief['attitude_score']}",
        f"- subjective_norm_score: {belief['subjective_norm_score']}",
        f"- pbc_score: {belief['pbc_score']}",
        f"- fertility_intention: {belief['fertility_intention_dist']}",
        "",
        compile_enumerate(memory_lines, header="Retrieved memories (most relevant first)")
        if memory_lines else "Retrieved memories: none",
    ]
    if new_messages:
        parts += ["", compile_enumerate(new_messages, header="New messages read this week")]
    parts += ["", "Output the updated TPB state and reflection_memory as JSON."]
    return BELIEF_UPDATE_SYSTEM, "\n".join(parts)


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
