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
import time

import requests
from dotenv import load_dotenv

from utils.logging_utils import get_logger
from utils.utils import parse_json_response

load_dotenv()

logger = get_logger("llm")

DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


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
            if not base:
                raise RuntimeError("Set LOCAL_LLM_URL in .env (or use LLM_PROVIDER=openrouter)")
            self.url = base.rstrip("/") + "/chat/completions"
            self.api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
            self.model = model or os.getenv("LOCAL_LLM_MODEL", "llama3")
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider!r} (use 'openrouter' or 'local')")

    def chat(self, system, user, temperature=None):
        """Send one system+user exchange and return the assistant text."""
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
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, json=payload, headers=headers,
                                     timeout=self.timeout)
                if resp.status_code == 429:
                    # rate limited: honour Retry-After if given, else back off hard
                    wait = float(resp.headers.get("Retry-After")
                                 or self.retry_wait * 4 * (attempt + 1))
                    last_error = f"429 Too Many Requests (waited {wait:.0f}s)"
                    logger.warning(f"429 rate limited (attempt {attempt + 1}/"
                                   f"{self.max_retries}), waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
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

    def chat_json(self, system, user, temperature=None):
        """chat() + JSON parsing, retrying once with a stricter reminder on parse failure."""
        text = self.chat(system, user, temperature=temperature)
        try:
            return parse_json_response(text)
        except ValueError:
            logger.warning(f"JSON parse failed, retrying with strict reminder. "
                           f"Raw response: {text[:500]!r}")
            text = self.chat(
                system,
                user + "\n\nIMPORTANT: Respond with VALID JSON ONLY. No prose, no markdown.",
                temperature=0.0,
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
