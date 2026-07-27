"""VacSim-recsys variant: engine_direct's no-TPB mechanics + VacSim's actual
embedding-similarity recommenders for news/tweet EXPOSURE (~/VacSim:
engines/engine.py's broadcast_news_and_policies/feed_tweets, via
src/recommenders/).

engines.engine_direct already replicates VacSim's memory-formation, retrieval
and single-call update mechanics faithfully; the one thing it deliberately
did NOT port was VacSim's recommenders -- it broadcasts the same fixed weekly
news item to every agent, and feeds an agent literally everything posted by
whoever they follow. This subclass changes only WHICH news/posts each agent
sees each week, by overriding DirectSimulation's two exposure-gathering hooks
(`_gather_news_lessons` / `_gather_social_lessons`); the gate, retrieval,
update-call and posting steps are inherited unchanged.

News: this project's per-condition `news_schedule` (sandbox.news) still picks
WHAT gets published each week and in what deterministic order (so runs stay
reproducible and comparable to the fixed-schedule direct runs) -- but instead
of broadcasting that week's single item to everyone, each week's item(s) are
added to a cumulative, growing public pool, and every agent is shown their
own top-`news_top_k` most relevant articles from EVERYTHING published so far
(recommenders.NewsRecommender, ranked by embedding similarity to the agent's
profile + most recent tweet).

Tweets: instead of "every post from someone I follow, unranked", each agent
is shown their own top-`tweet_top_k` tweets from the WHOLE population's tweet
pool so far, ranked by similarity to their own last tweet plus a following
boost (recommenders.TweetRecommender) -- a followed edge boosts rank, it no
longer gates visibility.
"""

import json
import os

from engines.engine import CONDITIONS
from engines.engine_direct import DirectSimulation, _parse_lessons
from recommenders.news_recommender import NewsRecommender
from recommenders.recommender import Recommender
from recommenders.tweet_recommender import TweetRecommender
from sandbox.prompts_direct import build_news_lessons_prompt, build_social_lessons_prompt

DEFAULT_NEWS_TOP_K = 3
DEFAULT_TWEET_TOP_K = 3
DEFAULT_FOLLOW_ALPHA = 0.3  # VacSim's TweetRecommender default


class RecsysDirectSimulation(DirectSimulation):
    """DirectSimulation with recommender-selected news/social exposure."""

    def __init__(self, *args, news_top_k=DEFAULT_NEWS_TOP_K,
                 tweet_top_k=DEFAULT_TWEET_TOP_K, follow_alpha=DEFAULT_FOLLOW_ALPHA,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.news_top_k = news_top_k
        self.tweet_top_k = tweet_top_k
        embed = Recommender()  # shared embedding cache across both recommenders
        self.news_recommender = NewsRecommender(embed)
        self.tweet_recommender = TweetRecommender(embed, self.network, alpha=follow_alpha)
        self._news_pool = []  # cumulative revealed articles, grows week by week
        self._week_news_recs = {}   # agent_id -> [article dict, ...] this week
        self._week_tweet_recs = {}  # agent_id -> [(author_id, text), ...] this week
        self.logger.info(f"RecsysDirectSimulation: news_top_k={news_top_k}, "
                         f"tweet_top_k={tweet_top_k}, follow_alpha={follow_alpha}")

    def _reveal_week_news(self, timestep):
        """Add this week's scheduled item(s) (policy and/or context, per
        sandbox.news.build_news_schedule) to the cumulative public pool."""
        for n in self.news_schedule.get(timestep, []):
            self._news_pool.append({
                "text": n.text, "news_id": n.news_id,
                "policy_name": n.policy.name, "revealed_at": timestep,
            })
        return self._news_pool

    def step(self, timestep):
        """Compute this week's per-agent recommendations ONCE (bulk, before
        any per-agent LLM calls), then run the normal weekly loop -- the
        gathering hooks below just read the precomputed dicts."""
        policy_on, social_on = CONDITIONS[self.condition]
        self._week_news_recs = (
            self.news_recommender.recommend(
                self._reveal_week_news(timestep), self.agents, self.news_top_k)
            if policy_on else {})
        self._week_tweet_recs = (
            self.tweet_recommender.recommend(self.agents, self.tweet_top_k)
            if social_on else {})
        super().step(timestep)

    # ── overridden exposure hooks (gate/retrieval/update/post inherited) ────

    def _gather_news_lessons(self, agent, timestep, news_items):
        recommended = self._week_news_recs.get(agent.agent_id, [])
        if not recommended:
            return []
        news_texts = [a["text"] for a in recommended]
        sys_n, usr_n = build_news_lessons_prompt(agent, news_texts)
        out = self._chat_json(sys_n, usr_n)
        return _parse_lessons(out, agent.agent_id, timestep, "policy_news")

    def _gather_social_lessons(self, agent, timestep, social_on):
        if not social_on:
            return []
        recommended = self._week_tweet_recs.get(agent.agent_id, [])
        if not recommended:
            return []
        post_texts = [text for _, text in recommended]
        sys_s, usr_s = build_social_lessons_prompt(agent, post_texts)
        out = self._chat_json(sys_s, usr_s)
        return _parse_lessons(out, agent.agent_id, timestep, "social_post")

    # ── persistence ───────────────────────────────────────────────────────

    def save(self):
        """Same shape as DirectSimulation.save(), plus `recsys_config` so a
        run is traceable to the top-k/alpha it was generated with.
        `methodology="direct_recsys"` distinguishes it from both the TPB
        study and the fixed-schedule direct replication at a glance."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{self.run_name}.json")
        state = {
            "condition": self.condition,
            "methodology": "direct_recsys",
            "recsys_config": {
                "news_top_k": self.news_top_k,
                "tweet_top_k": self.tweet_top_k,
                "follow_alpha": self.tweet_recommender.alpha,
            },
            "current_timestep": self.current_timestep,
            "network": self.network,
            "agents": [a.to_dict() for a in self.agents],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return path
