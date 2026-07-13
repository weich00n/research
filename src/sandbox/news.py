"""Policy news delivered to agents in conditions C2 and C3.

A News item is the message text an agent actually reads; the underlying
Policy object stays attached for traceability. With a pre-generated corpus
(generate_news_corpus.py, VacSim-style) each week serves a distinct article;
without one, the legacy announce-then-remind texts are used. Design:
outputs/analysis/news_dissemination_design.md.
"""

import itertools
import json
import random

from sandbox.policy import get_policies

RANDOM_STATE = 42

_id_counter = itertools.count(1)


class News:
    """A weekly policy news item shown to agents in conditions C2/C3.

    `text` is the message the agent actually reads (defaults to a sentence built
    from the policy name + description). The full `policy` object is kept for
    traceability; `to_dict` only serialises its name/category to stay JSON-safe.
    `article_type` is set for corpus articles (announcement / explainer /
    family_impact / roundup), None for legacy generated texts.
    """

    def __init__(self, policy, timestep, text=None, news_id=None, article_type=None):
        self.news_id = news_id or f"N{next(_id_counter):04d}"
        self.policy = policy
        self.timestep = timestep
        self.article_type = article_type
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
            "article_type": self.article_type,
            "text": self.text,
        }

    def __repr__(self):
        return f"News({self.news_id}, t={self.timestep}, {self.policy.name!r})"


def load_news_corpus(path):
    """Read a generate_news_corpus.py output file -> {policy_name: [articles]}.

    Accepts the full corpus dict (with metadata) or a bare article list.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"] if isinstance(data, dict) else data
    by_policy = {}
    for a in articles:
        by_policy.setdefault(a["policy_name"], []).append(a)
    return by_policy


def _build_variant_queues(corpus, seed):
    """Per-policy serving order: the announcement first (a policy's first week
    in a run reads as its announcement), then the ongoing-coverage variants
    (explainer / family_impact / roundup) in a seeded shuffled order, then any
    spare announcements last. Sorted by news_id before shuffling so the draw
    is reproducible regardless of corpus file order."""
    rng = random.Random(seed)
    queues = {}
    for name in sorted(corpus):
        entries = sorted(corpus[name], key=lambda a: a["news_id"])
        announcements = [a for a in entries if a["article_type"] == "announcement"]
        ongoing = [a for a in entries if a["article_type"] != "announcement"]
        rng.shuffle(ongoing)
        queues[name] = announcements[:1] + ongoing + announcements[1:]
    return queues


def build_news_schedule(num_timesteps, category=None, start_timestep=1,
                        corpus_path=None, seed=RANDOM_STATE):
    """Map timestep -> list[News], cycling through the chosen policy scenario.

    One policy news item is broadcast to all agents each week, in a fixed
    deterministic order (round-robin over the scenario's policies) so runs
    are reproducible. `category` of None uses the combined scenario.

    With `corpus_path` (a generate_news_corpus.py output), each week serves
    the next unused pre-generated article for that week's policy — no text
    repeats within a run, and repeat coverage arrives as ongoing-coverage
    article types instead of re-announcements (the anti-ratchet design). If a
    policy has no articles left (or none at all), it falls back to the legacy
    reminder text below.

    Without a corpus (default), behaviour is unchanged: after every policy has
    been announced once, later cycles are framed as reminders rather than
    fresh announcements — re-"announcing" a policy the agent already knows
    reads as new evidence every week and ratchets beliefs monotonically
    toward the scale ceiling.
    """
    policies = get_policies(category)
    if not policies:
        raise ValueError(f"No policies for category {category!r}")
    queues = _build_variant_queues(load_news_corpus(corpus_path), seed) \
        if corpus_path else {}
    schedule = {}
    for i, t in enumerate(range(start_timestep, start_timestep + num_timesteps)):
        policy = policies[i % len(policies)]
        text, news_id, article_type = None, None, None
        queue = queues.get(policy.name)
        if queue:
            entry = queue.pop(0)
            text = entry["text"]
            news_id = entry["news_id"]
            article_type = entry["article_type"]
        elif i >= len(policies):  # legacy repeat cycle: remind, don't re-announce
            text = (
                f"News this week: a reminder of the {policy.name}, which the "
                f"Singapore Government announced earlier. {policy.description}"
            )
        schedule[t] = [News(policy, t, text=text, news_id=news_id,
                            article_type=article_type)]
    return schedule
