"""TPB relevance scoring for memories (VacSim: LLM_judge.py).

Three interchangeable modes:
- "llm":    LLM-as-judge using the CLAUDE.md construct prompts.
- "cosine": embedding cosine similarity between the memory text and each
            construct embed anchor (needs sentence-transformers).
- "hybrid": cosine as a cheap prefilter (recall) + LLM judge as the reranker
            (precision). At creation a memory stores only its cosine prefilter
            scores; at retrieval `rerank()` shortlists the top-K per construct
            (union) and LLM-judges just those, cached. Needs both an LLM and an
            embedder.

All per-memory scores are {"attitude": float, "norm": float, "pbc": float} in [0, 1].
"""

from sandbox.prompts import CONSTRUCT_EMBED_PROMPTS, build_relevance_prompt
from utils.utils import clamp


class RelevanceScorer:
    """Scores how relevant a memory is to each TPB construct, in three modes.

    mode="llm" asks an LLM to judge (needs an LLMClient). mode="cosine" compares
    sentence embeddings of the memory against the (recall-broadened) construct embed
    anchors (needs sentence-transformers). mode="hybrid" uses cosine to shortlist
    candidates and the LLM to rerank them (needs both). In cosine/hybrid modes the
    embed anchors are embedded once up front and cached in `_construct_embeddings`.
    """

    def __init__(self, mode="llm", llm=None, embedder=None):
        if mode not in ("llm", "cosine", "hybrid"):
            raise ValueError(f"mode must be 'llm', 'cosine' or 'hybrid', got {mode!r}")
        self.mode = mode
        self.llm = llm
        self.embedder = embedder
        self._construct_embeddings = None
        if mode in ("llm", "hybrid") and llm is None:
            raise ValueError(f"mode={mode!r} requires an LLMClient")
        if mode in ("cosine", "hybrid"):
            if embedder is None:
                from utils.generate_utils import EmbeddingClient
                self.embedder = EmbeddingClient()
            keys = list(CONSTRUCT_EMBED_PROMPTS)
            vecs = self.embedder.embed([CONSTRUCT_EMBED_PROMPTS[k] for k in keys])
            self._construct_embeddings = dict(zip(keys, vecs))

    def score(self, memory_text):
        """Return {"attitude": x, "norm": y, "pbc": z}, each in [0,1], for one memory."""
        if self.mode == "llm":
            return self._score_llm(memory_text)
        return self._score_cosine(memory_text)

    def creation_scores(self, memory_text):
        """Scores to store on a NEW memory, as (relevance, cosine_relevance).

        - llm:    (LLM-judge scores, {})  — relevance is final at creation.
        - cosine: (cosine scores, {})     — relevance is final at creation.
        - hybrid: ({}, cosine scores)     — relevance left empty and filled lazily by
          `rerank()` at retrieval; cosine_relevance is the prefilter signal.
        """
        if self.mode == "llm":
            return self._score_llm(memory_text), {}
        if self.mode == "cosine":
            return self._score_cosine(memory_text), {}
        # hybrid
        return {}, self._score_cosine(memory_text)

    def rerank(self, lessons, top_k=12):
        """Hybrid retrieval-time rerank: cosine-shortlist, then LLM-judge the shortlist.

        For each TPB construct take the top_k simulation memories by their cosine
        prefilter score, then UNION those shortlists into one candidate set — a memory
        enters if it is top_k on ANY construct, which maximises recall into the pool
        (important for subjective_norm, which cosine underranks). Each candidate not
        yet judged gets one LLM-judge call, cached on `lesson.relevance` so it is
        judged at most once over the run. No-op unless mode='hybrid'.
        """
        if self.mode != "hybrid":
            return
        sim = [l for l in lessons if getattr(l, "memory_class", None) == "simulation"]
        if not sim:
            return
        candidates = set()
        for construct in self._construct_embeddings:
            ranked = sorted(sim, key=lambda l: l.cosine_relevance.get(construct, 0.0),
                            reverse=True)
            for l in ranked[:top_k]:
                if l.cosine_relevance.get(construct, 0.0) > 0.0:
                    candidates.add(l)
        for l in candidates:
            if not l.relevance:  # judge once, then cache
                l.relevance = self._score_llm(l.memory_text)

    def _score_llm(self, memory_text):
        """LLM-as-judge: ask for the three relevance scores; clamp each to [0,1]."""
        system, user = build_relevance_prompt(memory_text)
        out = self.llm.chat_json(system, user, temperature=0.0)
        # The judge is asked for a JSON object; if the model returns something
        # else (e.g. a bare list), fall back to zero relevance rather than
        # crashing on .get — a malformed judgement just means "not retrieved".
        if not isinstance(out, dict):
            out = {}
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
