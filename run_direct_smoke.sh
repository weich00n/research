#!/bin/bash
# Smoke-test the VacSim-direct (no-TPB) replication (driver_direct.py /
# engines/engine_direct.py) across all four conditions, on a small slice of
# agents/weeks so it's cheap to run and rerun. Uses the SAME seeded agents,
# news/context corpus, and social network as the TPB study (run_next_batch.sh)
# -- only the methodology differs: no TPB layer, weekly-batched memory
# formation, VacSim-style importance+decay retrieval (see CLAUDE.md /
# engines/engine_direct.py docstring for the full comparison).
#
# Usage:
#   tmux new -s fark_direct_smoke
#   bash run_direct_smoke.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_direct_smoke
#
# Override the defaults via env vars if you want a bigger/smaller smoke test,
# e.g.  NUM_AGENTS=20 TIMESTEPS=6 bash run_direct_smoke.sh
#
# Each run is checkpointed per-week and this script always passes --resume,
# so a dropped tmux/SSH session or a rerun just picks up where it left off;
# already-finished runs are skipped (see run() below). Output goes under
# outputs/runs/smoke/ (CLAUDE.md: throwaway smoke runs, not the tracked
# production runs), namespaced smoke_direct_C0..C3 so they never collide with
# the TPB study's run_C0..C3 or a full-scale run_C0..C3_direct.
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
  python driver_direct.py --run-name "$name" --agents "$AGENTS" \
    --num-agents "$NUM_AGENTS" --timesteps "$TIMESTEPS" \
    --concurrency "$CONCURRENCY" --output-dir "$RUNS_DIR" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $name ====="
}

# C0 — static baseline: no policy, no social. Just exercises seed generation +
# the t=0 baseline call; every week after that gates (no new inputs) and
# carries the intention forward untouched.
run smoke_direct_C0 --condition C0

# C1 — social only: exercises the batched social-lesson call + tweet posting
# + the network read path.
run smoke_direct_C1 --condition C1 --network "$NETWORK"

# C2 — policy only: exercises the batched news-lesson call against the same
# news corpus the TPB study uses.
run smoke_direct_C2 --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category combined

# C3 — policy + social: exercises everything together.
run smoke_direct_C3 --condition C3 --network "$NETWORK" --news-corpus "$NEWS_CORPUS"

echo
echo "All 4 smoke runs complete. Outputs in $RUNS_DIR/{smoke_direct_C0,smoke_direct_C1,smoke_direct_C2,smoke_direct_C3}.json"
