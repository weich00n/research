"""Small shared helpers for the sandbox package."""

import json
import re


def compile_enumerate(items, header=None):
    """Render a list of items as a numbered block of text for prompts."""
    lines = []
    if header:
        lines.append(f"{header}:")
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. {item}")
    return "\n".join(lines)


def parse_json_response(text):
    """Parse a JSON object/array out of an LLM response.

    Tolerates markdown code fences and leading/trailing chatter around the JSON.
    Raises ValueError if no JSON can be recovered.
    """
    text = text.strip()
    # 1. If the model wrapped the JSON in a ```json … ``` code fence, take the
    #    contents of the fence and ignore everything else.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 2. Try parsing the (possibly de-fenced) text as-is.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 3. Last resort: grab the widest {...} or [...] span and parse that. We use
    #    find() for the first opener and rfind() for the last closer, so leading
    #    /trailing chatter ("Sure, here's the JSON:") is sliced off.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def clamp(value, lo, hi):
    """Force `value` into the closed range [lo, hi]."""
    return max(lo, min(hi, value))


def normalise_distribution(dist, n=5):
    """Coerce a fertility-intention distribution to n non-negative floats summing to 1.0."""
    # Wrong length is a hard error — the LLM was asked for exactly n probabilities.
    if not isinstance(dist, (list, tuple)) or len(dist) != n:
        raise ValueError(f"Expected a list of {n} floats, got: {dist!r}")
    # Floor negatives at 0 (a probability can't be negative), then renormalise.
    vals = [max(0.0, float(v)) for v in dist]
    total = sum(vals)
    # Edge case: if the LLM returned all zeros/negatives there is nothing to
    # normalise, so fall back to a uniform distribution rather than dividing by 0.
    if total <= 0:
        return [1.0 / n] * n
    return [round(v / total, 4) for v in vals]
