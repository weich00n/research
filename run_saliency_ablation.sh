#!/bin/bash
# Retrieval-mechanism ablation (driver.py --retrieval saliency): re-runs the
# core C0-C3 arms with memory retrieval swapped from the CLAUDE.md formula
# (saliency x per-construct TPB-relevance) to a construct-blind alternative
# (retrieve_memories_saliency_only: pure saliency ranking, no relevance term
# at all). Same seeded agent file, network, and news corpus as the canonical
# TPB study, so C0-C3 here are directly comparable to run_C0_Qwen /
# run_C1_Qwen_fixed / run_C2_financial+caregiving / run_C3_Qwen.
#
# Purpose: the canonical study's TPB<->intention mediation result could in
# principle be an artifact of retrieval pre-filtering memories toward each
# construct's own relevance score (real circularity risk flagged in
# docs/agent_memory/tpb-construct-validity-plan.md). If TPB scores still
# move directionally and construct-delta still correlates with
# intention-delta when retrieval can't see TPB relevance at all, that's
# evidence the mediation is content-driven, not retrieval-driven.
#
# Both LLM calls (7a TPB update, 7b intention update) still see the SAME
# retrieved memory set here, same as the canonical engine — only WHICH
# memories get retrieved changes. Relevance scores are still computed and
# stored on every memory at creation (unused for selection in this mode) so
# post-hoc analysis can check what the TPB-relevance formula would have
# picked instead.
#
# Run this INSIDE tmux on the GPU server (checkpointed per-week, --resume
# always passed, safe to rerun after an interruption).
#
# Usage:
#   tmux new -s fark_saliency_ablation
#   bash run_saliency_ablation.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_saliency_ablation
#
# Requires: vLLM already serving Qwen2.5-14B (see serve_qwen.sh) and .env
# resolving to LLM_PROVIDER=local -- this script force-exports that so it can
# never silently fall back to OpenRouter (see preflight check below).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/src"

export LLM_PROVIDER=local
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

AGENTS="../agents_final_100_seeded.json"
NEWS_CORPUS="../outputs/news/news_corpus_qwen.json"
NETWORK="../outputs/networks/social_network_qwen_noarea.json"
TIMESTEPS="${TIMESTEPS:-12}"
CONCURRENCY="${CONCURRENCY:-32}"
RUNS_DIR="../outputs/runs"

# --- preflight: fail loudly instead of silently hitting OpenRouter ---
echo "Preflight: checking LLM provider resolves to 'local'..."
PROVIDER="$(python -c "from utils.generate_utils import LLMClient; print(LLMClient().provider)")"
if [ "$PROVIDER" != "local" ]; then
  echo "ERROR: LLM provider resolved to '$PROVIDER', expected 'local'." >&2
  echo "Check .env / LOCAL_LLM_URL(S) and that vLLM (serve_qwen.sh) is running." >&2
  exit 1
fi
echo "  OK: provider=local model=$LOCAL_LLM_MODEL"

run() {
  local name="$1"; shift
  local run_json="$RUNS_DIR/${name}.json"

  if [ -f "$run_json" ]; then
    local done_week
    done_week="$(python -c "import json; print(json.load(open('$run_json'))['current_timestep'])" 2>/dev/null || echo -1)"
    if [ "$done_week" = "$TIMESTEPS" ]; then
      echo
      echo "===== $name already complete (week $done_week/$TIMESTEPS) — skipping ====="
      return 0
    fi
  fi

  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  starting $name ====="
  python driver.py --run-name "$name" --agents "$AGENTS" --retrieval saliency \
    --timesteps "$TIMESTEPS" --concurrency "$CONCURRENCY" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $name ====="
}

run run_C0_saliencyretr --condition C0

run run_C1_saliencyretr --condition C1 --network "$NETWORK"

run run_C2_saliencyretr --condition C2 \
  --news-corpus "$NEWS_CORPUS" --policy-category combined

run run_C3_saliencyretr --condition C3 --network "$NETWORK" \
  --news-corpus "$NEWS_CORPUS" --policy-category combined

echo
echo "All 4 runs complete. Outputs in $RUNS_DIR/{run_C0_saliencyretr,run_C1_saliencyretr,run_C2_saliencyretr,run_C3_saliencyretr}.json"

echo
echo "===== Analyzing: does TPB<->intention mediation survive construct-blind retrieval? ====="
python validation/analyze_retrieval_ablation.py
