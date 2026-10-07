#!/bin/bash
# DOSE-MATCHED comparison via OpenRouter — the no-GPU route, hard-capped at $8.
#
# Same experiment as run_dose_matched.sh (see that file for why it exists), but
# against OpenRouter instead of the school vLLM, because the GPU box is full and
# the paper deadline is 16 Nov. Estimated cost ~$5 at Qwen2.5-14B's $0.20/M in
# and out; the cap is set to $8 for headroom.
#
# WHY ALL FOUR ARMS RUN HERE, not just the new one: the comparison must be
# internally consistent. An OpenRouter caregiving arm against a local-GPU
# financial arm would confound serving stack (and any quantization difference)
# with the treatment, which is exactly the kind of confound this experiment
# exists to remove. Financial is therefore re-run here too, and these four runs
# are compared only against each other.
#
# TWO INDEPENDENT SPEND CAPS — use both:
#   1. A credit limit on the OpenRouter KEY itself, set at openrouter.ai/keys.
#      This is the real cap: it holds even if this process is killed, the
#      accounting drifts, or something re-runs by accident.
#   2. LLM_BUDGET_USD below, enforced in utils/generate_utils.py:SpendGuard
#      against the token counts the API actually returns. Aborts the run before
#      the next call once the cap is passed. Engines checkpoint per completed
#      week, so --resume picks up cleanly afterwards.
#
# Usage:
#   export OPENROUTER_API_KEY=...        # a key with its own credit limit set
#   bash scripts/run_dose_matched_openrouter.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/src"

export LLM_PROVIDER=openrouter
# Verify this slug at openrouter.ai/models before a real run — provider slugs
# change, and a wrong one fails on the first call (cheaply, thanks to preflight).
export OPENROUTER_MODEL="${OPENROUTER_MODEL:-qwen/qwen-2.5-14b-instruct}"
export LLM_BUDGET_USD="${LLM_BUDGET_USD:-8}"
export LLM_PRICE_IN_PER_M="${LLM_PRICE_IN_PER_M:-0.20}"
export LLM_PRICE_OUT_PER_M="${LLM_PRICE_OUT_PER_M:-0.20}"

AGENTS="../agents_final_100_seeded.json"
NEWS_CORPUS="../outputs/news/news_corpus_qwen.json"
TIMESTEPS="${TIMESTEPS:-12}"
# Lower than the local default of 32: OpenRouter rate-limits harder than a
# dedicated vLLM box, and a 429 storm wastes wall-clock on backoff.
CONCURRENCY="${CONCURRENCY:-8}"
RUNS_DIR="../outputs/runs"
REPLICATES="${REPLICATES:-2}"
SUFFIX="${SUFFIX:-or}"   # keeps these runs distinct from any local-GPU ones

CAREGIVING_THREE=("Flexible Work Arrangement Request Guidelines" \
                  "Enhanced Paternity Leave" \
                  "Preschool & Infant Care Subsidies")

echo "Preflight:"
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "  ERROR: OPENROUTER_API_KEY unset" >&2; exit 1; }
[ -f "$AGENTS" ] || { echo "  ERROR: missing $AGENTS" >&2; exit 1; }
[ -f "$NEWS_CORPUS" ] || { echo "  ERROR: missing $NEWS_CORPUS" >&2; exit 1; }

# One real call: proves the model slug resolves AND measures tokens-per-call, so
# the cost projection is measured rather than guessed. Costs a fraction of a cent.
python - <<'PY' || exit 1
import json, os, sys
sys.path.insert(0, ".")
from utils.generate_utils import LLMClient, SPEND_GUARD
d = json.load(open("../outputs/news/news_corpus_qwen.json"))
arts = d["articles"] if isinstance(d, dict) else d
if len(arts) != 45:
    print(f"  ERROR: corpus has {len(arts)} articles, expected 45", file=sys.stderr); sys.exit(1)
print(f"  OK: corpus parses, {len(arts)} articles")
c = LLMClient()
print(f"  model: {c.provider} / {c.model}")
try:
    c.chat("Reply with the single word OK.", "Say OK.")
except Exception as e:
    print(f"  ERROR: test call failed -- check OPENROUTER_MODEL slug. {e}", file=sys.stderr)
    sys.exit(1)
g = SPEND_GUARD
tok = g.prompt_tokens + g.completion_tokens
print(f"  OK: test call succeeded ({tok} tokens, ${g.cost:.5f})")
# 4 runs x 100 agents x (12 weeks x 2 calls + 1 baseline) = ~10,000 calls.
# Real prompts are far larger than the test call, so project from the project's
# own measured ceiling (~2.3k prompt tokens) rather than from this probe.
est = 10_000 * (2300 * float(os.environ["LLM_PRICE_IN_PER_M"])
                + 250 * float(os.environ["LLM_PRICE_OUT_PER_M"])) / 1e6
print(f"  projected total: ~${est:.2f} against a ${float(os.environ['LLM_BUDGET_USD']):.2f} cap")
if est > float(os.environ["LLM_BUDGET_USD"]):
    print("  ERROR: projection exceeds the cap. Raise LLM_BUDGET_USD deliberately "
          "or cut REPLICATES.", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "$RUNS_DIR"

run() {
  local name="$1"; shift
  local run_json="$RUNS_DIR/${name}.json"
  if [ -f "$run_json" ]; then
    local done_week
    done_week="$(python -c "import json; print(json.load(open('$run_json'))['current_timestep'])" 2>/dev/null || echo -1)"
    if [ "$done_week" = "$TIMESTEPS" ]; then
      echo; echo "===== $name already complete — skipping ====="; return 0
    fi
  fi
  echo; echo "===== $(date '+%F %T')  starting $name ====="
  python driver_direct.py --run-name "$name" --agents "$AGENTS" \
    --condition C2 --news-corpus "$NEWS_CORPUS" \
    --timesteps "$TIMESTEPS" --concurrency "$CONCURRENCY" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%F %T')  finished $name ====="
}

for s in $(seq 1 "$REPLICATES"); do
  run "run_C2_direct_financial_dose3_${SUFFIX}_s$s"  --policy-category financial
  run "run_C2_direct_caregiving_dose3_${SUFFIX}_s$s" --policy-category caregiving \
      --policy-names "${CAREGIVING_THREE[@]}"
done

echo
echo "All four arms complete. Compare these to EACH OTHER only, not to the"
echo "local-GPU runs -- different serving stack."
