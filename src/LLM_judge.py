"""TPB relevance scoring for memories (VacSim: LLM_judge.py).

Two interchangeable modes:
- "llm":    LLM-as-judge using the CLAUDE.md construct prompts.
- "cosine": embedding cosine similarity between the memory text and each
            construct prompt (needs sentence-transformers).

Both return {"attitude": float, "norm": float, "pbc": float} in [0, 1].
"""

from sandbox.prompts import CONSTRUCT_PROMPTS, build_relevance_prompt
from utils.utils import clamp


class RelevanceScorer:
    """Scores how relevant a memory is to each TPB construct, in two modes.

    mode="llm" asks an LLM to judge (needs an LLMClient). mode="cosine" compares
    sentence embeddings of the memory against the construct prompts (needs
    sentence-transformers). In cosine mode the construct prompts are embedded
    once up front and cached in `_construct_embeddings`.
    """

    def __init__(self, mode="llm", llm=None, embedder=None):
        if mode not in ("llm", "cosine"):
            raise ValueError(f"mode must be 'llm' or 'cosine', got {mode!r}")
        self.mode = mode
        self.llm = llm
        self.embedder = embedder
        self._construct_embeddings = None
        if mode == "llm" and llm is None:
            raise ValueError("mode='llm' requires an LLMClient")
        if mode == "cosine":
            if embedder is None:
                from utils.generate_utils import EmbeddingClient
                self.embedder = EmbeddingClient()
            keys = list(CONSTRUCT_PROMPTS)
            vecs = self.embedder.embed([CONSTRUCT_PROMPTS[k] for k in keys])
            self._construct_embeddings = dict(zip(keys, vecs))

    def score(self, memory_text):
        """Return {"attitude": x, "norm": y, "pbc": z}, each in [0,1], for one memory."""
        if self.mode == "llm":
            return self._score_llm(memory_text)
        return self._score_cosine(memory_text)

    def _score_llm(self, memory_text):
        """LLM-as-judge: ask for the three relevance scores; clamp each to [0,1]."""
        system, user = build_relevance_prompt(memory_text)
        out = self.llm.chat_json(system, user, temperature=0.0)
        return {
            "attitude": clamp(float(out.get("attitude_relevance", 0.0)), 0.0, 1.0),
            "norm": clamp(float(out.get("norm_relevance", 0.0)), 0.0, 1.0),
            "pbc": clamp(float(out.get("pbc_relevance", 0.0)), 0.0, 1.0),
        }

    def _score_cosine(self, memory_text):
        """Embedding similarity: cosine(memory, each construct prompt), clipped to [0,1]."""
        vec = self.embedder.embed([memory_text])[0]
        scores = {}
        for construct, cvec in self._construct_embeddings.items():
            # embeddings are L2-normalised, so the dot product is cosine
            # similarity in [-1, 1]; clip to [0, 1] as the relevance score
            scores[construct] = clamp(float(vec @ cvec), 0.0, 1.0)
        return scores
