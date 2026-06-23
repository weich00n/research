"""Lesson = one memory in an agent's memory stream (CLAUDE.md Layer 3).

Keeps VacSim's "lesson" naming but carries the full CLAUDE.md memory schema.
Saliency decays exponentially: saliency = importance * LAMBDA^(current_t - created_t).
"""

import itertools

# Weekly decay factor for saliency. 0.995 ≈ a 0.5% drop per week, so memories
# fade slowly: a memory keeps ~78% of its strength after a year (52 weeks).
LAMBDA = 0.995

# Allowed values for the two categorical fields (asserted in __init__).
SOURCE_TYPES = ("profile_seed", "policy_news", "social_post", "reflection", "conversation")
MEMORY_CLASSES = ("seed", "simulation")
TPB_CONSTRUCTS = ("attitude", "norm", "pbc")

# Process-wide counter that hands out unique memory ids: M000001, M000002, …
_id_counter = itertools.count(1)


class Lesson:
    """One memory in an agent's memory stream (CLAUDE.md Layer 3).

    Carries the memory text plus the metadata retrieval needs: how important it
    is (`importance`), how relevant it is to each TPB construct (`relevance`),
    and when it was created (drives saliency decay).
    """

    def __init__(self, agent_id, memory_text, created_timestep, source_type,
                 importance, memory_class, source_agent_id=None,
                 available_from_timestep=None, relevance=None, memory_id=None):
        assert source_type in SOURCE_TYPES, f"bad source_type: {source_type}"
        assert memory_class in MEMORY_CLASSES, f"bad memory_class: {memory_class}"
        self.memory_id = memory_id or f"M{next(_id_counter):06d}"
        self.agent_id = agent_id
        self.created_timestep = created_timestep
        self.memory_text = memory_text
        self.source_type = source_type
        self.source_agent_id = source_agent_id
        self.importance = max(0.0, min(1.0, float(importance)))
        # posts become visible to others only from t+1
        self.available_from_timestep = (
            available_from_timestep if available_from_timestep is not None
            else created_timestep
        )
        self.memory_class = memory_class
        # {"attitude": float, "norm": float, "pbc": float}, each in [0, 1]
        self.relevance = relevance or {}
        self.retrieval_history = []
        self.used_in_update = False

    def saliency(self, current_timestep):
        """How 'loud' this memory is now: importance discounted by age (decay)."""
        return self.importance * (LAMBDA ** (current_timestep - self.created_timestep))

    def record_retrieval(self, timestep, construct):
        """Log that this memory was retrieved at `timestep` for a given TPB construct."""
        self.retrieval_history.append({"timestep": timestep, "construct": construct})

    def to_dict(self):
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "created_timestep": self.created_timestep,
            "memory_text": self.memory_text,
            "source_type": self.source_type,
            "source_agent_id": self.source_agent_id,
            "importance": self.importance,
            "available_from_timestep": self.available_from_timestep,
            "memory_class": self.memory_class,
            "relevance": self.relevance,
            "retrieval_history": self.retrieval_history,
            "used_in_update": self.used_in_update,
        }

    @classmethod
    def from_dict(cls, d):
        lesson = cls(
            agent_id=d["agent_id"],
            memory_text=d["memory_text"],
            created_timestep=d["created_timestep"],
            source_type=d["source_type"],
            importance=d["importance"],
            memory_class=d["memory_class"],
            source_agent_id=d.get("source_agent_id"),
            available_from_timestep=d.get("available_from_timestep"),
            relevance=d.get("relevance"),
            memory_id=d.get("memory_id"),
        )
        lesson.retrieval_history = d.get("retrieval_history", [])
        lesson.used_in_update = d.get("used_in_update", False)
        return lesson

    def __repr__(self):
        return (f"Lesson({self.memory_id}, t={self.created_timestep}, "
                f"{self.source_type}, imp={self.importance:.2f}, "
                f"{self.memory_text[:60]!r})")


def retrieve_memories(lessons, current_timestep, max_seed=5, per_construct=5, cap=20):
    """CLAUDE.md retrieval: up to `max_seed` seed memories (by saliency) plus the
    top `per_construct` simulation memories per TPB construct ranked by
    retrieval_score = normalised_saliency * TPB_relevance. No duplicates,
    at most `cap` memories in total.

    Returns (retrieved_lessons, construct_map) where construct_map records which
    construct(s) each memory was retrieved for.
    """
    # Split the memory stream into its two classes. Seed memories (stable
    # background beliefs) and simulation memories (things that happened during
    # the run) are retrieved by different rules.
    seed = [l for l in lessons if l.memory_class == "seed"]
    sim = [l for l in lessons if l.memory_class == "simulation"]

    # --- Seed memories: just take the `max_seed` most salient ones. ----------
    seed.sort(key=lambda l: l.saliency(current_timestep), reverse=True)
    selected = seed[:max_seed]
    # construct_map maps memory_id -> list of the reason(s) it was picked.
    # It doubles as the de-dup set: a memory already in it is never added twice.
    construct_map = {l.memory_id: ["seed"] for l in selected}
    for l in selected:
        l.record_retrieval(current_timestep, "seed")

    # --- Simulation memories: rank per TPB construct. ------------------------
    if sim:
        # Current saliency of every simulation memory.
        saliencies = {l.memory_id: l.saliency(current_timestep) for l in sim}
        # Min-max normalise saliency to [0, 1] so it is on the same scale as the
        # [0, 1] relevance scores before we multiply them. If every memory has
        # the same saliency (hi == lo) we treat them all as fully salient (1.0)
        # to avoid dividing by zero.
        lo, hi = min(saliencies.values()), max(saliencies.values())
        norm_sal = {mid: (s - lo) / (hi - lo) if hi > lo else 1.0
                    for mid, s in saliencies.items()}

        # Do one ranked pass per construct (attitude, norm, pbc).
        for construct in TPB_CONSTRUCTS:
            # retrieval_score = normalised_saliency × this construct's relevance;
            # sort all sim memories best-first by that score.
            scored = sorted(
                sim,
                key=lambda l: norm_sal[l.memory_id] * l.relevance.get(construct, 0.0),
                reverse=True,
            )
            # Consider the top `per_construct` for this construct.
            for l in scored[:per_construct]:
                # Skip memories with zero score (irrelevant or fully decayed) —
                # they would be noise.
                if norm_sal[l.memory_id] * l.relevance.get(construct, 0.0) <= 0:
                    continue
                if l.memory_id in construct_map:
                    # Already selected (by seed or another construct): just note
                    # this construct too, don't add a duplicate.
                    construct_map[l.memory_id].append(construct)
                    l.record_retrieval(current_timestep, construct)
                elif len(construct_map) < cap:
                    # New memory and we are still under the overall cap of 20.
                    selected.append(l)
                    construct_map[l.memory_id] = [construct]
                    l.record_retrieval(current_timestep, construct)

    return selected, construct_map
