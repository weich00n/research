"""Configurable LLM client (VacSim: utils/generate_utils.py).

Reads provider settings from .env so the same simulation code runs against
OpenRouter or any local OpenAI-compatible endpoint (llama.cpp, Ollama, vLLM):

    LLM_PROVIDER=openrouter | local        (default: openrouter)
    OPENROUTER_API_KEY=...                 (openrouter)
    OPENROUTER_MODEL=...                   (default: meta-llama/llama-3.3-70b-instruct:free)
    LOCAL_LLM_URL=http://localhost:8000/v1 (local, OpenAI-compatible base URL)
    LOCAL_LLM_MODEL=...                    (local model name)
"""

import os
import threading
import time

import requests
from dotenv import load_dotenv

from utils.logging_utils import get_logger
from utils.utils import parse_json_response

load_dotenv()

logger = get_logger("llm")

DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


class BudgetExceeded(RuntimeError):
    """Raised when cumulative paid-API spend passes LLM_BUDGET_USD.

    Safe to raise mid-run: the engines checkpoint per completed week and abort a
    week without a partial save, so `--resume` restarts cleanly at the last
    finished week.
    """


class SpendGuard:
    """Tracks paid-API spend from the `usage` block the provider returns and
    stops the run before it passes a cap.

    Inert unless LLM_BUDGET_USD is set, so local-vLLM runs are untouched. Counts
    the tokens the API actually billed rather than estimating from prompt length,
    and is thread-safe because the engines run agent-weeks through a pool.

    This is a SECOND line of defence. The real hard cap is a credit limit on the
    OpenRouter key itself, which holds even if this process is killed or the
    accounting drifts.
    """

    def __init__(self):
        self.budget = float(os.getenv("LLM_BUDGET_USD", "0") or 0)
        self.price_in = float(os.getenv("LLM_PRICE_IN_PER_M", "0") or 0)
        self.price_out = float(os.getenv("LLM_PRICE_OUT_PER_M", "0") or 0)
        self.enabled = self.budget > 0 and (self.price_in > 0 or self.price_out > 0)
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._warned = set()

    @property
    def cost(self):
        return (self.prompt_tokens / 1e6 * self.price_in
                + self.completion_tokens / 1e6 * self.price_out)

    def record(self, usage):
        """Add one call's usage; raise BudgetExceeded once the cap is passed."""
        if not self.enabled:
            return
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            cost = self.cost
            for frac in (0.5, 0.8, 0.95):
                if cost >= self.budget * frac and frac not in self._warned:
                    self._warned.add(frac)
                    logger.warning(f"LLM spend at {frac:.0%} of budget: "
                                   f"${cost:.2f} / ${self.budget:.2f} "
                                   f"after {self.calls} calls")
            if cost >= self.budget:
                raise BudgetExceeded(
                    f"spend ${cost:.2f} reached the ${self.budget:.2f} cap after "
                    f"{self.calls} calls ({self.prompt_tokens:,} prompt + "
                    f"{self.completion_tokens:,} completion tokens). Run aborted "
                    f"before the next call. Re-run with --resume to continue from "
                    f"the last completed week, or raise LLM_BUDGET_USD.")

    def check(self):
        """Raise BEFORE issuing a request if the cap is already passed.

        Without this the engines' per-agent retry loop would bill several more
        calls after the cap, since record() only runs once a response is back.
        """
        if not self.enabled:
            return
        with self._lock:
            if self.cost >= self.budget:
                raise BudgetExceeded(
                    f"spend ${self.cost:.2f} is at or past the ${self.budget:.2f} "
                    f"cap; refusing further calls. Re-run with --resume after "
                    f"raising LLM_BUDGET_USD.")

    def summary(self):
        if not self.enabled:
            return f"{self.calls} calls (no budget cap set)"
        return (f"{self.calls} calls, {self.prompt_tokens:,} prompt + "
                f"{self.completion_tokens:,} completion tokens, "
                f"${self.cost:.2f} of ${self.budget:.2f} budget")


# Process-wide, so every LLMClient in a run shares one budget rather than each
# getting its own allowance.
SPEND_GUARD = SpendGuard()


class LLMClient:
    """Talks to one chat-completions endpoint (OpenRouter or local), with retries.

    The provider, URL, key and model are resolved from `.env` at construction
    (see the module docstring). `chat()` returns raw text; `chat_json()` parses
    the response into a Python object. Both retry on transient failures.
    """

    def __init__(self, provider=None, model=None, temperature=0.7,
                 max_retries=6, retry_wait=5, timeout=120):
        # Provider is chosen by the `provider` arg, else LLM_PROVIDER in .env,
        # else "openrouter". Each branch below validates that its required env
        # vars exist and raises a helpful error if not.
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openrouter")).lower()
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.timeout = timeout

        if self.provider == "openrouter":
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise RuntimeError("Set OPENROUTER_API_KEY in .env (or use LLM_PROVIDER=local)")
            self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        elif self.provider == "local":
            base = os.getenv("LOCAL_LLM_URL")
            # LOCAL_LLM_URLS (comma-separated) lets one client fan requests across
            # several vLLM servers (one per GPU) when --data-parallel-size isn't
            # available; otherwise a single LOCAL_LLM_URL (e.g. a DP endpoint) is used.
            urls_env = os.getenv("LOCAL_LLM_URLS")
            if urls_env:
                self.urls = [u.strip().rstrip("/") + "/chat/completions"
                             for u in urls_env.split(",") if u.strip()]
                self.url = self.urls[0]
            elif base:
                self.url = base.rstrip("/") + "/chat/completions"
            else:
                raise RuntimeError("Set LOCAL_LLM_URL or LOCAL_LLM_URLS in .env "
                                   "(or use LLM_PROVIDER=openrouter)")
            self.api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
            self.model = model or os.getenv("LOCAL_LLM_MODEL", "llama3")
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider!r} (use 'openrouter' or 'local')")

        # All endpoints this client may target. Single-element unless LOCAL_LLM_URLS
        # set multiple; the engine round-robins over it for concurrent runs.
        self.urls = getattr(self, "urls", None) or [self.url]

    def chat(self, system, user, temperature=None, url=None):
        """Send one system+user exchange and return the assistant text.

        `url` overrides the default endpoint (used to spread concurrent calls
        across several local vLLM servers); defaults to self.url.
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error = None
        start = time.time()
        SPEND_GUARD.check()  # fail before spending, not after
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url or self.url, json=payload, headers=headers,
                                     timeout=self.timeout)
                if resp.status_code == 429:
                    # rate limited: honour Retry-After if given, else back off hard.
                    # RFC 7231 also allows Retry-After to be an HTTP-date string,
                    # which float() can't parse -- fall back to the hard backoff
                    # rather than letting a ValueError crash the whole run.
                    try:
                        wait = float(resp.headers.get("Retry-After")
                                     or self.retry_wait * 4 * (attempt + 1))
                    except (ValueError, TypeError):
                        wait = self.retry_wait * 4 * (attempt + 1)
                    last_error = f"429 Too Many Requests (waited {wait:.0f}s)"
                    logger.warning(f"429 rate limited (attempt {attempt + 1}/"
                                   f"{self.max_retries}), waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                body = resp.json()
                # Record before the empty-content check: an empty completion is
                # still billed, so it must count against the cap.
                SPEND_GUARD.record(body.get("usage") or {})
                content = body["choices"][0]["message"]["content"]
                # Reasoning models occasionally return null/empty content (the
                # budget went to the hidden reasoning trace). Treat as transient
                # and retry rather than crashing downstream on len(None).
                if not content:
                    last_error = "empty/None content in response"
                    logger.warning(f"empty content (attempt {attempt + 1}/"
                                   f"{self.max_retries}), retrying")
                    time.sleep(self.retry_wait * (attempt + 1))
                    continue
                logger.debug(f"{self.model} ok in {time.time() - start:.1f}s "
                             f"(attempt {attempt + 1}, prompt {len(system) + len(user)} "
                             f"chars, response {len(content)} chars)")
                return content
            except (requests.RequestException, KeyError, IndexError) as e:
                # Network errors (RequestException) and malformed responses
                # (KeyError/IndexError when digging into the JSON) are both
                # treated as transient: back off linearly and retry.
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/"
                               f"{self.max_retries}): {e}")
                time.sleep(self.retry_wait * (attempt + 1))
        logger.error(f"LLM call gave up after {self.max_retries} attempts: {last_error}")
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def chat_json(self, system, user, temperature=None, url=None):
        """chat() + JSON parsing, retrying once with a stricter reminder on parse failure."""
        text = self.chat(system, user, temperature=temperature, url=url)
        try:
            return parse_json_response(text)
        except ValueError:
            logger.warning(f"JSON parse failed, retrying with strict reminder. "
                           f"Raw response: {text[:500]!r}")
            text = self.chat(
                system,
                user + "\n\nIMPORTANT: Respond with VALID JSON ONLY. No prose, no markdown.",
                temperature=0.0,
                url=url,
            )
            return parse_json_response(text)


class EmbeddingClient:
    """Sentence-transformer embeddings for cosine-similarity TPB relevance.

    Loaded lazily so the LLM-as-judge path has no extra dependency.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Cosine relevance mode needs sentence-transformers: "
                "pip install sentence-transformers"
            ) from e
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
