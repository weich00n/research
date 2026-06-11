"""Fertility-related social post (VacSim's Tweet).

Posted at timestep t, visible to connected agents from t+1.
"""

import itertools

_id_counter = itertools.count(1)


class Tweet:
    def __init__(self, agent_id, text, created_timestep, tweet_id=None):
        self.tweet_id = tweet_id or f"T{next(_id_counter):06d}"
        self.agent_id = agent_id
        self.text = text
        self.created_timestep = created_timestep
        self.available_from_timestep = created_timestep + 1

    def visible_at(self, timestep):
        return timestep >= self.available_from_timestep

    def to_dict(self):
        return {
            "tweet_id": self.tweet_id,
            "agent_id": self.agent_id,
            "text": self.text,
            "created_timestep": self.created_timestep,
            "available_from_timestep": self.available_from_timestep,
        }

    @classmethod
    def from_dict(cls, d):
        tweet = cls(d["agent_id"], d["text"], d["created_timestep"],
                    tweet_id=d.get("tweet_id"))
        return tweet

    def __repr__(self):
        return f"Tweet({self.tweet_id}, {self.agent_id}, t={self.created_timestep}, {self.text[:60]!r})"
