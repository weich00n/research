"""Personalised social feed (~/VacSim: src/recommenders/tweet_recommender.py).

VacSim ranks candidate tweets for an agent by cosine similarity to that
agent's own most recent tweet, plus an additive `alpha`-weighted boost when
the tweet's author is someone the agent follows (`agent.following: {id:
weight}`). Two adaptations for this project:

- This project's social network (`utils.network_utils`) is a plain directed
  adjacency `{agent_id: [followed_ids]}`, not VacSim's weighted dict -- every
  edge here counts as weight 1.0, so the boost is a flat `alpha` for any
  followed author (VacSim's own worked example is a full-strength edge, i.e.
  the same number).
- VacSim requires the agent to already have a tweet of its own before it can
  recommend anything (`assert recent_tweets[0] != None`) -- a hard
  precondition its always-tweeting agents satisfy from day 2 onward. Week 1
  here has literally no tweets in the whole population yet (nobody has
  posted), so instead of asserting, an agent with no tweet of their own falls
  back to a pure following-weighted ranking ("what did people I follow say"),
  which degrades gracefully to VacSim's steady-state behaviour once agents
  start posting.

Ranks over ALL tweets posted so far by ALL agents (not just followed ones) --
this is the actual difference from the plain "read my followed agents' feed"
mechanic in `engines.engine_direct`: a followed edge boosts rank, it doesn't
gate visibility.
"""

import numpy as np


class TweetRecommender:
    def __init__(self, recommender, network, alpha=0.3):
        self.r = recommender
        self.network = network  # {agent_id: [followed_ids]}
        self.alpha = alpha

    def recommend(self, agents, num_recommendations=3):
        """Returns {agent_id: [(author_id, tweet_text), ...]}, up to
        `num_recommendations` per agent, highest-scoring first. Never
        recommends an agent their own tweets."""
        all_tweets = [(a.agent_id, t.text) for a in agents for t in a.tweets]
        out = {a.agent_id: [] for a in agents}
        if not all_tweets:
            return out

        tweet_vecs = self.r.encode([text for _, text in all_tweets])
        for agent in agents:
            candidate_idx = [i for i, (author, _) in enumerate(all_tweets)
                             if author != agent.agent_id]
            if not candidate_idx:
                continue
            following = set(self.network.get(agent.agent_id, []))
            boost = np.array([self.alpha if all_tweets[i][0] in following else 0.0
                              for i in candidate_idx])
            if agent.tweets:
                own_vec = self.r.encode_one(agent.tweets[-1].text)
                sim = tweet_vecs[candidate_idx] @ own_vec  # normalised -> cosine
            else:
                sim = np.zeros(len(candidate_idx))
            score = sim + boost
            k = min(num_recommendations, len(candidate_idx))
            ranked = [candidate_idx[j] for j in score.argsort()[::-1][:k]]
            out[agent.agent_id] = [all_tweets[i] for i in ranked]
        return out
