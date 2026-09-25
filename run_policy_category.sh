#!/bin/bash
# EXPERIMENT 1 — policy category on the no-TPB engine.
#
# Question: does the financial-vs-caregiving intention difference survive when
# the agent has no TPB belief layer? The headline result (caregiving +0.212 vs
# financial +0.103, paired t=5.23, p=9.5e-7, dz=0.52) comes from the TPB engine
# (run_C2_{financial,caregiving} + _rep2). A reviewer can ask whether the
# saturating construct layer leaked into the intention call, since both calls
# read the same retrieved memories. This answers that directly: if caregiving
# still beats financial with no belief layer, the ranking is not an artifact of
# the architecture.
#
# Design: C2 only (policy news, no social) so peer contagion cannot confound the
# category contrast. Two replicates per category. The news schedule is seeded
# (RANDOM_STATE=42) and no driver exposes --seed, so replicates differ ONLY by
# LLM sampling at temperature 0.7 -- that is the intended control, since it holds
# the stimulus constant. Replicate spread therefore measures sampling variance,
# not robustness to a different policy ordering.
#
# 4 runs x 100 agents x 12 weeks, ~20 min each, ~80 min total.
#
# IMPORTANT when analysing: the TPB and direct engines establish t=0 differently
# BY DESIGN (see docs/agent_memory -> baseline-confound note). Compare DELTAS,
# never raw endpoints, across engines.
#
# Usage (on the GPU server, inside tmux):
#   tmux new -s policy_category
#   bash run_policy_category.sh
#   # detach: Ctrl-b d ; reattach: tmux attach -t policy_category
#
# Requires vLLM already serving Qwen2.5-14B (serve_qwen.sh) and .env resolving
# to LLM_PROVIDER=local -- forced below so it can never silently hit OpenRouter.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/src"

export LLM_PROVIDER=local
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

AGENTS="../agents_final_100_seeded.json"
NEWS_CORPUS="../outputs/news/news_corpus_qwen.json"
TIMESTEPS="${TIMESTEPS:-12}"
CONCURRENCY="${CONCURRENCY:-32}"
RUNS_DIR="../outputs/runs"
REPLICATES="${REPLICATES:-2}"

# --- preflight -------------------------------------------------------------
echo "Preflight:"

if [ ! -f "$AGENTS" ]; then
  echo "  ERROR: missing $AGENTS" >&2; exit 1
fi
if [ ! -f "$NEWS_CORPUS" ]; then
  echo "  ERROR: missing $NEWS_CORPUS" >&2
  echo "  outputs/ is gitignored, so the corpus does NOT arrive via git pull." >&2
  echo "  Copy it from the laptop before running." >&2
  exit 1
fi
echo "  OK: agents + news corpus present"

PROVIDER="$(python -c "from utils.generate_utils import LLMClient; print(LLMClient().provider)")"
if [ "$PROVIDER" != "local" ]; then
  echo "  ERROR: provider resolved to '$PROVIDER', expected 'local'." >&2
  echo "  Check .env / LOCAL_LLM_URL(S) and that vLLM is running." >&2
  exit 1
fi
echo "  OK: provider=local model=$LOCAL_LLM_MODEL"
mkdir -p "$RUNS_DIR"

# --- runner ----------------------------------------------------------------
run() {
  local name="$1"; shift
  local run_json="$RUNS_DIR/${name}.json"

  if [ -f "$run_json" ]; then
    local done_week
    done_week="$(python -c "import json; print(json.load(open('$run_json'))['current_timestep'])" 2>/dev/null || echo -1)"
    if [ "$done_week" = "$TIMESTEPS" ]; then
      echo; echo "===== $name already complete (week $done_week/$TIMESTEPS) — skipping ====="
      return 0
    fi
  fi

  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  starting $name ====="
  python driver_direct.py --run-name "$name" --agents "$AGENTS" \
    --condition C2 --news-corpus "$NEWS_CORPUS" \
    --timesteps "$TIMESTEPS" --concurrency "$CONCURRENCY" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $name ====="
}

for s in $(seq 1 "$REPLICATES"); do
  run "run_C2_direct_financial_clean_s$s"  --policy-category financial
  run "run_C2_direct_caregiving_clean_s$s" --policy-category caregiving
done

echo
echo "All runs complete. Bring them back to the laptop with:"
echo "  scp 'USER@HOST:/data1/USER/research/outputs/runs/run_C2_direct_*_clean_s*.*' outputs/runs/"
