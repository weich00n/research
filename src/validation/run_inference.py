"""Run one rater model over the validation personas (resumable).

One JSONL output per model: outputs/validation/preds_<tag>.jsonl, one line
per persona with both inferences. Safe to interrupt and re-run — completed
personas are skipped, failed ones are re-attempted (the loader keeps the
last line per persona_id).

Prompt versions (see inference_prompts.PROMPT_SETS) are kept in separate
files: preds_<tag>_<version>.jsonl. Default is v2.

From src/:

    python validation/run_inference.py --model nemotron-120b --prompt-version v1 --limit 5
    python validation/run_inference.py --model nemotron-120b
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from utils.generate_utils import LLMClient
from utils.logging_utils import get_logger, setup_logger
from validation.inference_prompts import PROMPT_SETS

logger = get_logger("validation")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "..", "..", "outputs", "validation")
DEFAULT_PERSONAS = os.path.join(DEFAULT_OUT_DIR, "validation_personas_500.csv")

TEMPERATURE = 0.2  # matches the production notebook runs

MODELS = {
    "nemotron-120b": {"provider": "openrouter",
                      "model": "nvidia/nemotron-3-super-120b-a12b:free"},
    "gpt-4o-mini":   {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
    "gpt-4.1":       {"provider": "openrouter", "model": "openai/gpt-4.1"},
    "claude-haiku":  {"provider": "openrouter",
                      "model": "anthropic/claude-haiku-4.5"},
    "llama-3.1-8b":  {"provider": "local",
                      "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    "qwen2.5-14b":   {"provider": "local", "model": "Qwen/Qwen2.5-14B-Instruct"},
}


def load_done(path):
    """persona_ids already completed successfully (last line per id wins)."""
    records = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records[rec["persona_id"]] = rec
    return {pid for pid, rec in records.items() if not rec.get("error")}


def infer_persona(llm, row, prompts):
    """Both inferences for one persona. Returns (record_fields, error)."""
    out = {"fin_score": None, "fin_reasoning": None,
           "rel_status": None, "rel_reasoning": None}

    result = llm.chat_json(prompts["fin_system"], prompts["build_fin"](row),
                           temperature=TEMPERATURE)
    score = int(result["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"fin score out of range: {score}")
    out["fin_score"] = score
    out["fin_reasoning"] = str(result["reasoning"])

    if row["rel_status"] == "single":
        result = llm.chat_json(prompts["rel_system"], prompts["build_rel"](row),
                               temperature=TEMPERATURE)
        status = str(result["status"]).strip().title()
        if status not in {"Single", "Dating"}:
            raise ValueError(f"unexpected rel status: {status}")
        out["rel_status"] = status
        out["rel_reasoning"] = str(result["reasoning"])
    else:
        out["rel_status"] = "Married"  # pass-through, mirrors production

    return out


def main():
    parser = argparse.ArgumentParser(description="Validation inference runner")
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--prompt-version", default="v2",
                        choices=list(PROMPT_SETS))
    parser.add_argument("--personas", default=DEFAULT_PERSONAS)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N personas (smoke test)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model}_{args.prompt_version}"
    out_path = os.path.join(args.out_dir, f"preds_{tag}.jsonl")
    setup_logger(os.path.join(args.out_dir, f"run_{tag}.log"))

    cfg = MODELS[args.model]
    prompts = PROMPT_SETS[args.prompt_version]
    llm = LLMClient(provider=cfg["provider"], model=cfg["model"],
                    temperature=TEMPERATURE)
    logger.info(f"Rater {args.model}: {llm.provider} / {llm.model} "
                f"(prompts {args.prompt_version})")

    personas = pd.read_csv(args.personas)
    if args.limit:
        personas = personas.head(args.limit)

    done = load_done(out_path)
    todo = personas[~personas["persona_id"].isin(done)]
    logger.info(f"{len(personas)} personas, {len(done)} already done, "
                f"{len(todo)} to run")

    n_err = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for k, (_, row) in enumerate(todo.iterrows(), 1):
            rec = {"persona_id": row["persona_id"], "model_tag": args.model,
                   "model_id": cfg["model"],
                   "prompt_version": args.prompt_version, "error": None,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                rec.update(infer_persona(llm, row, prompts))
                logger.info(f"[{k}/{len(todo)}] {row['persona_id']} -> "
                            f"fin={rec['fin_score']} rel={rec['rel_status']}")
            except Exception as e:
                n_err += 1
                rec["error"] = str(e)
                logger.warning(f"[{k}/{len(todo)}] {row['persona_id']} FAILED: {e}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(f"Done: {len(todo) - n_err} ok, {n_err} failed -> {out_path}")
    if n_err:
        logger.info("Re-run the same command to retry the failures.")


if __name__ == "__main__":
    main()
