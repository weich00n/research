"""Policy news delivered to agents in conditions C2 and C3.

A News item is the message text an agent actually reads; the underlying
Policy object stays attached for traceability.
"""

import itertools

from sandbox.policy import get_policies

_id_counter = itertools.count(1)


class News:
    """A weekly policy news item shown to agents in conditions C2/C3.

    `text` is the message the agent actually reads (defaults to a sentence built
    from the policy name + description). The full `policy` object is kept for
    traceability; `to_dict` only serialises its name/category to stay JSON-safe.
    """

    def __init__(self, policy, timestep, text=None, news_id=None):
        self.news_id = news_id or f"N{next(_id_counter):04d}"
        self.policy = policy
        self.timestep = timestep
        self.text = text or (
            f"News this week: the Singapore Government announced the "
            f"{policy.name}. {policy.description}"
        )

    def to_dict(self):
        return {
            "news_id": self.news_id,
            "policy_name": self.policy.name,
            "policy_category": self.policy.category,
            "timestep": self.timestep,
            "text": self.text,
        }

    def __repr__(self):
        return f"News({self.news_id}, t={self.timestep}, {self.policy.name!r})"


def build_news_schedule(num_timesteps, category=None, start_timestep=1):
    """Map timestep -> list[News], cycling through the chosen policy scenario.

    One policy news item is broadcast to all agents each week, in a fixed
    deterministic order (round-robin over the scenario's policies) so runs
    are reproducible. `category` of None uses the combined scenario.
    """
    policies = get_policies(category)
    if not policies:
        raise ValueError(f"No policies for category {category!r}")
    schedule = {}
    for i, t in enumerate(range(start_timestep, start_timestep + num_timesteps)):
        schedule[t] = [News(policies[i % len(policies)], t)]
    return schedule
