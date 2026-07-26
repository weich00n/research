#!/bin/bash
# Full-scale VacSim-direct (no-TPB) batch (driver_direct.py / engine_direct.py):
# 100 agents, 12 weeks, same 7 arms validated by run_direct_smoke.sh +
# run_direct_arms_smoke.sh, at production scale. Run this INSIDE tmux on the
# GPU server overnight -- it's sequential and each run checkpoints per-week.
#
# Arms:
#   run_C0_direct                    static baseline (no policy, no social)
#   run_C1_direct                    social only
#   run_C2_direct                    policy only, just news (combined category)
#   run_C3_direct                    policy + social, just news
#   run_C3_direct_ambient_combined   policy + social, news + ambient context together
#   run_C3_direct_context_only       policy + social, ambient context only, no real policy (balanced)
#   run_C3_direct_context_headwind   policy + social, ambient context only, no real policy (negative / headwind)
#
# The last two omit --news-corpus on purpose: news.build_news_schedule only
# goes context-only when --context-corpus is set AND --news-corpus is absent
# (news.py: context_only = context_corpus_path is not None and corpus_path is
# None), and the engine only reads the schedule at all when policy_on=True --
# same trick run_next_batch.sh uses for run_C1_ambient/run_C3_ambient.
#
# Usage:
#   tmux new -s fark_direct_full
#   bash run_direct_full.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_direct_full
#
# Each driver_direct.py run is checkpointed per-week and this script always
# passes --resume, so if tmux/SSH drops overnight, just rerun this same
# script -- the in-progress run picks up from its last completed week and any
# already-finished runs are skipped (see run() below).
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
  python driver_direct.py --run-name "$name" --agents "$AGENTS" \
    --timesteps "$TIMESTEPS" --concurrency "$CONCURRENCY" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $name ====="
}

run run_C0_direct --condition C0

run run_C1_direct --condition C1 --network "$NETWORK"

run run_C2_direct --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category combined

run run_C3_direct --condition C3 --network "$NETWORK" \
  --news-corpus "$NEWS_CORPUS" --policy-category combined

run run_C3_direct_ambient_combined --condition C3 --network "$NETWORK" \
  --news-corpus "$NEWS_CORPUS" --policy-category combined \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C3_direct_context_only --condition C3 --network "$NETWORK" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C3_direct_context_headwind --condition C3 --network "$NETWORK" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix negative

echo
echo "All 7 runs complete. Outputs in $RUNS_DIR/{run_C0_direct,run_C1_direct,run_C2_direct,run_C3_direct,run_C3_direct_ambient_combined,run_C3_direct_context_only,run_C3_direct_context_headwind}.json"
