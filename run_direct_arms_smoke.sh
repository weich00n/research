#!/bin/bash
# Smoke-test the VacSim-direct (no-TPB) replication (driver_direct.py) across
# the news/context CONTENT axis, on a small slice of agents/weeks. Companion
# to run_direct_smoke.sh (which covers the four experimental CONDITIONS,
# C0-C3); this script instead varies what a policy-on condition actually
# delivers, all paired with C3 (policy+social, the fullest condition) so each
# arm also exercises the social/tweet path:
#
#   smoke_direct_C3_news_only        real policy news only, no ambient context
#   smoke_direct_C3_ambient_combined real policy news + ambient context together
#   smoke_direct_C3_context_only     ambient context only, no real policy (balanced valence)
#   smoke_direct_C3_context_headwind ambient context only, no real policy (negative valence)
#
# "context only" / "headwind" both rely on news.build_news_schedule's
# context_only branch (news.py: context_only = context_corpus_path is not
# None and corpus_path is None) -- so --news-corpus is deliberately OMITTED
# for those two arms and --context-corpus is set instead. This is the same
# trick run_next_batch.sh uses for run_C1_ambient/run_C3_ambient: the engine
# only reads the news schedule at all when policy_on=True, so getting
# "ambient context, no real policy" requires a policy-on condition (C2/C3)
# with --news-corpus omitted, not a naive C1 + --context-corpus (which the
# engine would never read).
#
# Usage:
#   tmux new -s fark_direct_arms_smoke
#   bash run_direct_arms_smoke.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_direct_arms_smoke
#
# Override the defaults via env vars, e.g. NUM_AGENTS=20 TIMESTEPS=6 bash ...
#
# Each run is checkpointed per-week and this script always passes --resume,
# so a dropped session or a rerun just picks up where it left off; already-
# finished runs are skipped (see run() below). Output goes under
# outputs/runs/smoke/ (throwaway, not tracked production runs).
#
# Requires: vLLM already serving Qwen2.5-14B (see serve_qwen.sh) and .env
# resolving to LLM_PROVIDER=local -- this script force-exports that so it
# can never silently fall back to OpenRouter (see preflight check below).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/src"

export LLM_PROVIDER=local
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

AGENTS="../agents_final_100_seeded.json"
NEWS_CORPUS="../outputs/news/news_corpus_qwen.json"
CONTEXT_CORPUS="../outputs/news/context_corpus_qwen.json"
NETWORK="../outputs/networks/social_network_qwen_noarea.json"
NUM_AGENTS="${NUM_AGENTS:-5}"
TIMESTEPS="${TIMESTEPS:-3}"
CONCURRENCY="${CONCURRENCY:-8}"
RUNS_DIR="../outputs/runs/smoke"

# --- preflight: fail loudly instead of silently hitting OpenRouter ---
echo "Preflight: checking LLM provider resolves to 'local'..."
PROVIDER="$(python -c "from utils.generate_utils import LLMClient; print(LLMClient().provider)")"
if [ "$PROVIDER" != "local" ]; then
  echo "ERROR: LLM provider resolved to '$PROVIDER', expected 'local'." >&2
  echo "Check .env / LOCAL_LLM_URL(S) and that vLLM (serve_qwen.sh) is running." >&2
  exit 1
fi
echo "  OK: provider=local model=$LOCAL_LLM_MODEL"

mkdir -p "$RUNS_DIR"

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
  python driver_direct.py --run-name "$name" --agents "$AGENTS" --condition C3 \
    --network "$NETWORK" --num-agents "$NUM_AGENTS" --timesteps "$TIMESTEPS" \
    --concurrency "$CONCURRENCY" --output-dir "$RUNS_DIR" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $name ====="
}

# Just news: real policy content, no ambient background.
run smoke_direct_C3_news_only \
  --news-corpus "$NEWS_CORPUS" --policy-category combined

# Combined: real policy news + ambient background together.
run smoke_direct_C3_ambient_combined \
  --news-corpus "$NEWS_CORPUS" --policy-category combined \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

# Context only: ambient background, no real policy (--news-corpus omitted).
run smoke_direct_C3_context_only \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

# Headwind: same context-only trick, pure negative-valence draw pool.
run smoke_direct_C3_context_headwind \
  --context-corpus "$CONTEXT_CORPUS" --context-mix negative

echo
echo "All 4 arm-comparison runs complete. Outputs in $RUNS_DIR/{smoke_direct_C3_news_only,smoke_direct_C3_ambient_combined,smoke_direct_C3_context_only,smoke_direct_C3_context_headwind}.json"
