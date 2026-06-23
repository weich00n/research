"""Fertility agent (CLAUDE.md Agent Structure, 3 layers).

Layer 1: static profile from agents_initialised.json (never changes).
Layer 2: dynamic TPB belief state (updates each timestep).
Layer 3: memory stream of Lesson objects (grows each timestep).
"""

from sandbox.lesson import Lesson, retrieve_memories
from sandbox.tweet import Tweet
from utils.utils import clamp, normalise_distribution

NEUTRAL_BELIEF_STATE = {
    "attitude_score": 3.0,
    "subjective_norm_score": 3.0,
    "pbc_score": 3.0,
    "fertility_intention_dist": None,  # set by the first LLM belief update
}


class Agent:
    """One simulated Singapore resident, stored as three stacked layers:

    Layer 1 (static profile) is read from the agent JSON and never changes.
    Layer 2 (`belief_state`) holds the three TPB scores + fertility-intention
    distribution and is overwritten each week by the LLM belief update.
    Layer 3 (`lessons`) is the memory stream, a growing list of Lesson objects.
    """

    def __init__(self, profile):
        # ── Layer 1: static profile ────────────────────────────────────────
        self.agent_id = profile["agent_id"]
        self.age = profile["age"]
        self.gender = profile["gender"]
        self.marital_status = profile["marital_status"]
        self.relationship_status = profile["relationship_status"]
        self.education = profile["education"]
        self.occupation = profile["occupation"]
        self.industry = profile["industry"]
        self.planning_area = profile["planning_area"]
        self.financial_security_score = profile["financial_security_score"]
        self.financial_security_reasoning = profile.get("financial_security_reasoning")
        self.general_persona = profile["general_persona"]
        self.cultural_background = profile["cultural_background"]
        self.hobbies_and_interests = profile["hobbies_and_interests"]
        self.career_goals = profile["career_goals"]
        self.source_index = profile.get("source_index")

        # ── Layer 2: dynamic belief state ──────────────────────────────────
        self.belief_state = dict(profile.get("belief_state") or NEUTRAL_BELIEF_STATE)
        self.belief_history = []  # [{"timestep": t, **belief_state}, ...]

        # ── Layer 3: memory stream ─────────────────────────────────────────
        self.lessons = [Lesson.from_dict(m) for m in profile.get("memory_stream", [])]

        self.tweets = []

    # ── Memory stream ──────────────────────────────────────────────────────

    def add_lesson(self, lesson):
        """Append a new memory (Lesson) to this agent's memory stream."""
        self.lessons.append(lesson)

    @property
    def seed_lessons(self):
        return [l for l in self.lessons if l.memory_class == "seed"]

    def retrieve(self, current_timestep):
        """CLAUDE.md retrieval: <=5 seed + <=5 per TPB construct, max 20, no dupes."""
        return retrieve_memories(self.lessons, current_timestep)

    # ── Belief state ───────────────────────────────────────────────────────

    def update_belief_state(self, attitude_score, subjective_norm_score, pbc_score,
                            fertility_intention, timestep):
        """Overwrite Layer 2 with the LLM's new TPB scores for this week.

        The LLM is free-form, so we defensively coerce its output:
        `clamp(..., 1.0, 5.0)` forces each score back onto the valid 1-5 TPB
        scale, and `normalise_distribution` repairs the intention list so it is
        5 non-negative floats summing to 1.0. The previous state is appended to
        `belief_history` first so the full weekly trajectory is preserved.
        """
        self.belief_state = {
            "attitude_score": clamp(float(attitude_score), 1.0, 5.0),
            "subjective_norm_score": clamp(float(subjective_norm_score), 1.0, 5.0),
            "pbc_score": clamp(float(pbc_score), 1.0, 5.0),
            "fertility_intention_dist": normalise_distribution(fertility_intention),
        }
        self.belief_history.append({"timestep": timestep, **self.belief_state})

    # ── Tweets ─────────────────────────────────────────────────────────────

    def post_tweet(self, text, timestep):
        """Create and store a social post by this agent (readable by followers at t+1)."""
        tweet = Tweet(self.agent_id, text, timestep)
        self.tweets.append(tweet)
        return tweet

    def tweets_visible_at(self, timestep):
        """Posts this agent's followers can read at `timestep` (posted at t-1)."""
        return [t for t in self.tweets if t.visible_at(timestep)]

    # ── Profile strings ────────────────────────────────────────────────────

    def get_profile_str(self):
        """Compact profile for social network generation (VacSim style).

        Demographics plus the one-sentence persona; the long narrative fields
        (hobbies, cultural background, career goals) are left out because all
        199 other agents' profiles share one prompt during network generation.
        The full profile for belief prompts is prompts.profile_to_str.
        """
        return (f"Gender: {self.gender}\tAge: {self.age}\t"
                f"Relationship: {self.relationship_status}\t"
                f"Education: {self.education}\tOccupation: {self.occupation}\t"
                f"Industry: {self.industry}\tArea: {self.planning_area}\t"
                f"Persona: {self.general_persona}")

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self):
        """Flatten all three layers into a JSON-serialisable dict (used by save())."""
        return {
            "agent_id": self.agent_id,
            "age": self.age,
            "gender": self.gender,
            "marital_status": self.marital_status,
            "relationship_status": self.relationship_status,
            "education": self.education,
            "occupation": self.occupation,
            "industry": self.industry,
            "planning_area": self.planning_area,
            "financial_security_score": self.financial_security_score,
            "financial_security_reasoning": self.financial_security_reasoning,
            "general_persona": self.general_persona,
            "cultural_background": self.cultural_background,
            "hobbies_and_interests": self.hobbies_and_interests,
            "career_goals": self.career_goals,
            "source_index": self.source_index,
            "belief_state": self.belief_state,
            "belief_history": self.belief_history,
            "memory_stream": [l.to_dict() for l in self.lessons],
            "tweets": [t.to_dict() for t in self.tweets],
        }

    def __repr__(self):
        return (f"Agent({self.agent_id}, {self.age}{self.gender[0]}, "
                f"{self.relationship_status}, {self.planning_area})")


def load_agents(path):
    """Load agents from agents_initialised.json."""
    import json
    with open(path, encoding="utf-8") as f:
        profiles = json.load(f)
    return [Agent(p) for p in profiles]
