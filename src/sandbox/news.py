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

from sandbox.policy import Policy, get_policies

RANDOM_STATE = 42

CONTEXT_MIXES = ("balanced", "negative", "positive", "neutral")


def context_policy(topic):
    """Stub Policy for ambient context articles (cost of living, housing,
    job market, ...): not a treatment instrument, no expected pathways."""
    return Policy(name=topic, category="context", description="",
                  expected_pathways=[])

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


def _build_context_queue(corpus_path, mix, seed):
    """Seeded, deterministic serving order for context articles.

    `mix` picks the draw pool: a pure valence ('negative'/'positive'/
    'neutral') or 'balanced' (all articles, natural corpus ratio). Sorted by
    news_id before shuffling so the order is reproducible regardless of file
    order; a separate RNG stream (seed+1) keeps the policy-article draw
    identical whether or not context is enabled.
    """
    if mix not in CONTEXT_MIXES:
        raise ValueError(f"context_mix must be one of {CONTEXT_MIXES}")
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"] if isinstance(data, dict) else data
    pool = sorted((a for a in articles
                   if mix == "balanced" or a.get("valence") == mix),
                  key=lambda a: a["news_id"])
    if not pool:
        raise ValueError(f"no context articles for mix {mix!r} in {corpus_path}")
    random.Random(seed + 1).shuffle(pool)
    return pool


def build_news_schedule(num_timesteps, category=None, start_timestep=1,
                        corpus_path=None, seed=RANDOM_STATE,
                        context_corpus_path=None, context_mix="balanced"):
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

    With `context_corpus_path` (generate_context_corpus.py output), one
    ambient context article is appended to each week's items, drawn without
    repetition from the pool selected by `context_mix` ('balanced' /
    'negative' / 'positive' / 'neutral'). **Context-only mode:** if
    `corpus_path` is None while `context_corpus_path` is given, the schedule
    contains context articles only (no policy items) — the no-policy
    background arm for the situation factorial.
    """
    context_only = context_corpus_path is not None and corpus_path is None
    policies = get_policies(category)
    if not policies:
        raise ValueError(f"No policies for category {category!r}")
    queues = _build_variant_queues(load_news_corpus(corpus_path), seed) \
        if corpus_path else {}
    context_queue = _build_context_queue(context_corpus_path, context_mix, seed) \
        if context_corpus_path else []
    schedule = {}
    for i, t in enumerate(range(start_timestep, start_timestep + num_timesteps)):
        items = []
        if not context_only:
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
            items.append(News(policy, t, text=text, news_id=news_id,
                              article_type=article_type))
        if context_queue:  # exhausted pool -> no context that week (never crash)
            entry = context_queue.pop(0)
            items.append(News(context_policy(entry["policy_name"]), t,
                              text=entry["text"], news_id=entry["news_id"],
                              article_type=entry["article_type"]))
        schedule[t] = items
    return schedule
