"""Lightweight agent for the VacSim-direct replication (no TPB layer).

Same static profile (Layer 1) as sandbox.agent.Agent, loaded from the same
agents_final_100_seeded.json, but Layer 2 is a single fertility_intention
distribution (no attitude/norm/pbc scores) and Layer 3 keeps only the seed
memories from that file -- everything else this run produces (weekly lessons,
the t=0 baseline, tweets) is built fresh by engines.engine_direct, so this
methodology never reads or writes the TPB study's belief_state/belief_history.
"""

import json

from sandbox.lesson import Lesson
from sandbox.tweet import Tweet
from utils.utils import normalise_distribution


class DirectAgent:
    """One simulated Singapore resident for the no-TPB (VacSim-direct) run.

    Layer 1 (static profile) mirrors sandbox.agent.Agent exactly, read from the
    same agent JSON. Layer 2 (`fertility_intention_dist` + `intention_history`)
    replaces the TPB belief_state/belief_history with a single distribution,
    updated by one call per week instead of two decoupled ones. Layer 3
    (`lessons`) starts from the seed memories only; the TPB run's simulation-
    phase memories (reflections, per-item perceived memories) are not carried
    over since this methodology forms memories differently (weekly batches).
    """

    def __init__(self, profile):
        # ── Layer 1: static profile (identical fields to sandbox.agent.Agent) ──
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

        # ── Layer 2: single intention distribution, no TPB constructs ────────
        self.fertility_intention_dist = profile.get("fertility_intention_dist")
        self.intention_history = list(profile.get("intention_history", []))

        # ── Layer 3: seed memories carried over; everything else built fresh ─
        self.lessons = [
            Lesson.from_dict(m) for m in profile.get("memory_stream", [])
            if m.get("memory_class") == "seed"
        ]
        self.tweets = [Tweet.from_dict(t) for t in profile.get("tweets", [])]

    def add_lesson(self, lesson):
        """Append a new memory (Lesson) to this agent's memory stream."""
        self.lessons.append(lesson)

    @property
    def seed_lessons(self):
        return [l for l in self.lessons if l.memory_class == "seed"]

    def update_intention(self, fertility_intention, timestep, reasoning=""):
        """Overwrite the intention distribution for this week (VacSim: one call,
        no separate belief-score update to decouple from)."""
        self.fertility_intention_dist = normalise_distribution(fertility_intention)
        self.intention_history.append({
            "timestep": timestep,
            "fertility_intention_dist": self.fertility_intention_dist,
            "reasoning": reasoning,
        })

    def post_tweet(self, text, timestep):
        tweet = Tweet(self.agent_id, text, timestep)
        self.tweets.append(tweet)
        return tweet

    def tweets_visible_at(self, timestep):
        return [t for t in self.tweets if t.visible_at(timestep)]

    def get_profile_str(self, include_area=True, include_persona=False):
        """Same rendering as sandbox.agent.Agent.get_profile_str (kept in sync
        for the shared social-network-generation script)."""
        s = (f"Gender: {self.gender}\tAge: {self.age}\t"
             f"Relationship: {self.relationship_status}\t"
             f"Education: {self.education}\tOccupation: {self.occupation}\t"
             f"Industry: {self.industry}")
        if include_area:
            s += f"\tArea: {self.planning_area}"
        if include_persona:
            s += f"\tAbout: {self.general_persona}"
        return s

    def to_dict(self):
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
            "fertility_intention_dist": self.fertility_intention_dist,
            "intention_history": self.intention_history,
            "memory_stream": [l.to_dict() for l in self.lessons],
            "tweets": [t.to_dict() for t in self.tweets],
        }

    def __repr__(self):
        return (f"DirectAgent({self.agent_id}, {self.age}{self.gender[0]}, "
                f"{self.relationship_status}, {self.planning_area})")


def load_agents(path):
    """Load DirectAgents from agents_final_100_seeded.json (or any agent JSON
    carrying the same fields; only static profile + seed memories are read)."""
    with open(path, encoding="utf-8") as f:
        profiles = json.load(f)
    return [DirectAgent(p) for p in profiles]
