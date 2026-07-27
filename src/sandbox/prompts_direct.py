"""Prompt templates for the VacSim-direct replication (no TPB layer).

Mirrors VacSim's own mechanics (~/VacSim: engines/engine.py, sandbox/prompts.py)
on this project's data instead of the TPB-mediation design in sandbox.prompts:

- Memories are formed in WEEKLY BATCHES -- all of a week's news in one call
  (VacSim's broadcast_news_and_policies), all of a week's followed-posts in
  another (VacSim's feed_tweets) -- not one perception call per item.
- No per-construct relevance tagging at all (see engines.engine_direct's
  importance+decay-only retrieval, VacSim's Lesson.score).
- The fertility_intention distribution is updated by a SINGLE call per week
  (VacSim's attitude_prompt) -- there is no TPB layer to decouple it from.

profile_to_str and build_seed_memory_prompt are reused as-is from
sandbox.prompts: both are already profile-only with no TPB framing, so they
apply unchanged to a DirectAgent.
"""

from sandbox.prompts import build_seed_memory_prompt, profile_to_str  # noqa: F401 (re-exported)
from utils.utils import compile_enumerate

MAX_LESSONS_PER_BATCH = 5

LESSON_JSON_INSTRUCTIONS = """Respond with JSON only:
{"lessons": [{"memory_text": "...", "importance": 0.7}, ... up to N items ...]}
Each memory_text is ONE short first-person takeaway. Rate importance on a 0-1
scale (0 = barely memorable, 1 = life-changing). Do not repeat near-duplicate
takeaways -- combine similar ones into a single lesson."""


# ──────────────────────────────────────────────────────────────────────────
# Weekly batched memory formation (VacSim: broadcast_news_and_policies / feed_tweets)
# ──────────────────────────────────────────────────────────────────────────

NEWS_LESSONS_SYSTEM = f"""You are simulating how a Singapore resident internalises
a batch of fertility/family-policy news they read this week. Given their
profile and the news items below, summarise the takeaways relevant to
marriage, family, career, finances, housing, or having children.

{LESSON_JSON_INSTRUCTIONS}"""


def build_news_lessons_prompt(agent, news_texts, max_lessons=MAX_LESSONS_PER_BATCH):
    """(system, user) prompt turning a WEEK's worth of news into <=max_lessons
    lessons in one call (all articles compiled into one prompt)."""
    items = compile_enumerate(news_texts, header="News read this week")
    user = (
        "Profile of the person:\n"
        f"{profile_to_str(agent)}\n\n"
        f"{items}\n\n"
        f"Summarise up to {max_lessons} takeaways as JSON."
    )
    return NEWS_LESSONS_SYSTEM, user


SOCIAL_LESSONS_SYSTEM = f"""You are simulating how a Singapore resident internalises
a batch of social-media posts from people they follow this week. Given their
profile and the posts below, summarise the takeaways relevant to marriage,
family, career, finances, housing, or having children.

{LESSON_JSON_INSTRUCTIONS}"""


def build_social_lessons_prompt(agent, post_texts, max_lessons=MAX_LESSONS_PER_BATCH):
    """(system, user) prompt turning a WEEK's worth of followed-agent posts into
    <=max_lessons lessons in one call (all posts compiled into one prompt)."""
    items = compile_enumerate(post_texts, header="Posts read this week")
    user = (
        "Profile of the person:\n"
        f"{profile_to_str(agent)}\n\n"
        f"{items}\n\n"
        f"Summarise up to {max_lessons} takeaways as JSON."
    )
    return SOCIAL_LESSONS_SYSTEM, user


# ──────────────────────────────────────────────────────────────────────────
# Fertility intention update -- ONE call per week (VacSim: attitude_prompt).
# No TPB scores exist in this methodology, so there is nothing to isolate the
# intention call from.
# ──────────────────────────────────────────────────────────────────────────

INTENTION_UPDATE_SYSTEM = """You are simulating the fertility intention of a
Singapore resident. Based on their profile, their previous fertility
intention, and the most important things they have learned so far (shown
below, most important first), output their updated fertility intention.

fertility_intention is a probability distribution over 5 ordinal intention
levels [p1, p2, p3, p4, p5] that sums to 1.0, where
1 = no child intention, 2 = weak/unlikely, 3 = uncertain, 4 = likely,
5 = strong intention.

Judge their intention holistically from their overall situation. Update
gradually and realistically: intention reflects their overall life
circumstances, which rarely change from week to week -- most weeks the right
update is no change or a very small one. You do not have to change your
intention just because you learned something new -- stay consistent with your
persona and only shift when it genuinely would. Briefly explain your
reasoning in one sentence.

Respond with JSON only:
{"reasoning": "...", "fertility_intention": [0.1, 0.2, 0.4, 0.2, 0.1]}"""


def _lesson_lines(retrieved_lessons):
    return [f"(t={l.created_timestep}, {l.source_type}) {l.memory_text}"
            for l in retrieved_lessons]


def build_intention_update_prompt(agent, retrieved_lessons, current_timestep):
    prev = agent.fertility_intention_dist
    prev_line = (
        f"Previous fertility intention distribution: {prev}"
        if prev is not None
        else "This is the first assessment of their fertility intention."
    )
    lines = _lesson_lines(retrieved_lessons)
    parts = [
        "Profile of the person:",
        profile_to_str(agent),
        "",
        f"Current week of the simulation: {current_timestep}",
        "",
        prev_line,
        "",
        compile_enumerate(lines, header="Most important things learned so far")
        if lines else "Nothing new learned so far.",
        "",
        "Output the updated fertility_intention distribution as JSON.",
    ]
    return INTENTION_UPDATE_SYSTEM, "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# t=0 baseline (VacSim: init_agents) -- from seed memories only, no TPB anchor.
# ──────────────────────────────────────────────────────────────────────────

BASELINE_INTENTION_SYSTEM = """You are establishing the INITIAL fertility
intention of a Singapore resident, based only on their profile and stable
background (seed) memories -- before the simulation begins (no policy or
social input yet).

fertility_intention is a probability distribution over 5 ordinal intention
levels [p1, p2, p3, p4, p5] that sums to 1.0, where
1 = no child intention, 2 = weak/unlikely, 3 = uncertain, 4 = likely,
5 = strong intention.

Judge their intention holistically from their overall situation. Use the FULL
range where warranted -- do NOT default to a flat or uniformly uncertain
distribution unless that genuinely fits this person. Briefly explain your
reasoning in one sentence.

Respond with JSON only:
{"reasoning": "...", "fertility_intention": [0.3, 0.3, 0.2, 0.1, 0.1]}"""


def build_baseline_intention_prompt(agent, retrieved_lessons):
    lines = _lesson_lines(retrieved_lessons)
    parts = [
        "Profile of the person:",
        profile_to_str(agent),
        "",
        "This is the BASELINE assessment, before the simulation begins "
        "(no policy or social input yet).",
        "",
        compile_enumerate(lines, header="Background memories")
        if lines else "No background memories.",
        "",
        "Output the baseline fertility_intention distribution as JSON.",
    ]
    return BASELINE_INTENTION_SYSTEM, "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Social post (tweet) generation -- same idea as sandbox.prompts.build_tweet_prompt
# but references the intention distribution instead of TPB scores.
# ──────────────────────────────────────────────────────────────────────────

TWEET_SYSTEM = """You are simulating a Singapore resident deciding whether to
post on social media this week. Based on their profile and current fertility
intention, decide if they would share something related to marriage, family,
parenting, children, or family-related policies. People do not post every
week; only post if this week genuinely gave them something to say.

If they post, write it in their authentic voice (casual Singapore
social-media tone, 1-3 sentences, no hashtag spam). Do not mention scores,
intention percentages, or the simulation.

Respond with JSON only:
{"post": true, "text": "..."}  or  {"post": false, "text": null}"""


def build_tweet_prompt(agent, reasoning_text, current_timestep):
    user = (
        "Profile of the person:\n"
        f"{profile_to_str(agent)}\n\n"
        f"Current week: {current_timestep}\n"
        f"Current fertility intention distribution: {agent.fertility_intention_dist}\n"
        f"This week's reasoning: \"{reasoning_text}\"\n\n"
        "Decide whether they post and output JSON."
    )
    return TWEET_SYSTEM, user
