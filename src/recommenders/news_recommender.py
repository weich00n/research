"""Personalised news selection (~/VacSim: src/recommenders/news_recommender.py).

VacSim recommends each agent `num_news` articles from that day's candidate
window by cosine similarity to the agent's OWN most recent tweet -- i.e. what
you get shown depends on what you've been saying, not a fixed broadcast. This
project's C0-C3 factorial has a condition (C2: policy-only, no social) where
agents never tweet by design, so "most recent tweet" isn't always available.
Adaptation: the recommendation basis is the agent's profile string, blended
with their most recent tweet's text when they have one (C1/C3) -- profile
similarity is the always-available fallback VacSim didn't need because its
agents always posted daily.

No stance/purity bookkeeping is ported (VacSim used it only for its own
console logging, not the recommendation itself).
"""


class NewsRecommender:
    """Wraps a `Recommender` (shared embedding cache) to rank a candidate news
    pool per agent."""

    def __init__(self, recommender):
        self.r = recommender

    def _agent_basis_text(self, agent):
        parts = [agent.get_profile_str(include_persona=True)]
        if agent.tweets:
            parts.append(agent.tweets[-1].text)
        return "\n".join(parts)

    def recommend(self, candidate_articles, agents, num_recommendations=3):
        """candidate_articles: list of dicts with a 'text' key (the cumulative
        pool of news revealed up to and including the current week).
        agents: list of DirectAgent-like objects.

        Returns {agent_id: [article dict, ...]}, up to `num_recommendations`
        per agent, ranked by cosine similarity (highest first). Deterministic
        given the same candidate pool and agent state (no sampling).
        """
        if not candidate_articles:
            return {a.agent_id: [] for a in agents}
        k = min(num_recommendations, len(candidate_articles))
        article_vecs = self.r.encode([a["text"] for a in candidate_articles])
        basis_vecs = self.r.encode([self._agent_basis_text(a) for a in agents])
        sims = basis_vecs @ article_vecs.T  # (n_agents, n_articles), normalised -> cosine
        out = {}
        for i, agent in enumerate(agents):
            top_idx = sims[i].argsort()[::-1][:k]
            out[agent.agent_id] = [candidate_articles[j] for j in top_idx]
        return out
