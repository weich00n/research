#!/bin/bash
# Full-scale VacSim-recsys batch (driver_direct_recsys.py /
# engine_direct_recsys.py): 100 agents, 12 weeks, same 10 arms as
# run_direct_full.sh, at production scale. Run this INSIDE tmux on the GPU
# server -- it's sequential and each run checkpoints per-week.
#
# Only difference from run_direct_full.sh: news/tweet EXPOSURE is
# recommender-selected per agent (embedding similarity, ported from
# ~/VacSim's src/recommenders/) instead of a fixed weekly broadcast / plain
# follow-list. Everything else (arms, corpora, network, gating, retrieval,
# the single update call) is identical -- see
# engines/engine_direct_recsys.py's module docstring.
#
# Arms:
#   run_C0_recsys                       static baseline (no policy, no social; recommenders never run)
#   run_C1_recsys                       social only
#   run_C2_recsys_news_only             policy only, just news (combined category)
#   run_C2_recsys_ambient_combined      policy only, news + ambient context together
#   run_C2_recsys_context_only          policy only, ambient context only, no real policy (balanced)
#   run_C2_recsys_context_headwind      policy only, ambient context only, no real policy (negative / headwind)
#   run_C3_recsys_news_only             policy + social, just news
#   run_C3_recsys_ambient_combined      policy + social, news + ambient context together
#   run_C3_recsys_context_only          policy + social, ambient context only, no real policy (balanced)
#   run_C3_recsys_context_headwind      policy + social, ambient context only, no real policy (negative / headwind)
#
# Usage:
#   tmux new -s fark_recsys_full
#   bash run_direct_recsys_full.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_recsys_full
#
# Each driver_direct_recsys.py run is checkpointed per-week and this script
# always passes --resume, so if a run errors out or tmux/SSH drops, just fix
# whatever broke and rerun this same script -- the in-progress run picks up
# from its last completed week and any already-finished runs are skipped
# (see run() below). This has NOT yet been exercised against a real LLM (only
# a stub-LLM dry run + unit-level recommender checks so far, see the
# vacsim-recsys branch commit) -- expect the first arm or two to surface
# integration bugs; that's what this script (not the smoke script) is for.
#
# Requires: vLLM already serving Qwen2.5-14B (see serve_qwen.sh) and .env
# resolving to LLM_PROVIDER=local -- this script force-exports that so it can
# never silently fall back to OpenRouter (see preflight check below). Also
# requires sentence-transformers (already a project dependency) -- the
# recommenders download all-MiniLM-L6-v2 on first use if not already cached.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/src"

export LLM_PROVIDER=local
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

AGENTS="../agents_final_100_seeded.json"
NEWS_CORPUS="../outputs/news/news_corpus_qwen.json"
CONTEXT_CORPUS="../outputs/news/context_corpus_qwen.json"
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
  python driver_direct_recsys.py --run-name "$name" --agents "$AGENTS" \
    --timesteps "$TIMESTEPS" --concurrency "$CONCURRENCY" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $name ====="
}

run run_C0_recsys --condition C0

run run_C1_recsys --condition C1 --network "$NETWORK"

# --- C2: policy only, all 4 content arms ---

run run_C2_recsys_news_only --condition C2 \
  --news-corpus "$NEWS_CORPUS" --policy-category combined

run run_C2_recsys_ambient_combined --condition C2 \
  --news-corpus "$NEWS_CORPUS" --policy-category combined \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C2_recsys_context_only --condition C2 \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C2_recsys_context_headwind --condition C2 \
  --context-corpus "$CONTEXT_CORPUS" --context-mix negative

# --- C3: policy + social, all 4 content arms ---

run run_C3_recsys_news_only --condition C3 --network "$NETWORK" \
  --news-corpus "$NEWS_CORPUS" --policy-category combined

run run_C3_recsys_ambient_combined --condition C3 --network "$NETWORK" \
  --news-corpus "$NEWS_CORPUS" --policy-category combined \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C3_recsys_context_only --condition C3 --network "$NETWORK" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C3_recsys_context_headwind --condition C3 --network "$NETWORK" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix negative

echo
echo "All 10 runs complete. Outputs in $RUNS_DIR/{run_C0_recsys,run_C1_recsys,run_C2_recsys_news_only,run_C2_recsys_ambient_combined,run_C2_recsys_context_only,run_C2_recsys_context_headwind,run_C3_recsys_news_only,run_C3_recsys_ambient_combined,run_C3_recsys_context_only,run_C3_recsys_context_headwind}.json"
