"""Base recommender (~/VacSim: src/recommenders/recommender.py).

VacSim's base class wraps a raw `SentenceTransformer` and hand-rolls cosine
similarity, plus incremental similarity-matrix caching across days (an
optimisation for its larger multi-day, many-agent runs). This port:

- Reuses this project's own `utils.generate_utils.EmbeddingClient`
  (sentence-transformers/all-MiniLM-L6-v2, `normalize_embeddings=True`) instead
  of instantiating a second `SentenceTransformer` — same model family VacSim
  used (`paraphrase-MiniLM-L6-v2`), and normalised embeddings make cosine
  similarity a plain dot product, matching the convention already used by
  `LLM_judge.py` / `validation/compare_runs.py`.
- Recomputes similarities from scratch each week rather than caching an
  incremental similarity matrix. At this project's scale (100 agents, 12
  weeks, small weekly candidate pools) that cache is an optimisation VacSim
  needed and this port doesn't -- recomputing is simpler and produces
  identical rankings.

Subclasses (`NewsRecommender`, `TweetRecommender`) each implement their own
`recommend(...)`.
"""

import numpy as np

from utils.generate_utils import EmbeddingClient


class Recommender:
    """Shared embedding cache: encode(text) -> normalised vector, memoised."""

    def __init__(self, embed_client=None):
        self.embed = embed_client or EmbeddingClient()
        self._cache = {}

    def encode(self, texts):
        """Embed a list of strings, reusing cached vectors where possible."""
        misses = [t for t in texts if t not in self._cache]
        if misses:
            vecs = self.embed.embed(misses)
            for t, v in zip(misses, vecs):
                self._cache[t] = v
        return np.array([self._cache[t] for t in texts])

    def encode_one(self, text):
        return self.encode([text])[0]
