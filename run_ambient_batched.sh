#!/bin/bash
# Remaining 2 of the 4 reruns needed for the batched-tweet-perception fix
# (test/batched-tweet-perception branch). run_C1_Qwen_batched and
# run_C3_Qwen_batched are already done (12/12 weeks) -- this script does the
# ambient-context counterparts, matching run_next_batch.sh's exact recipe
# for run_C1_ambient / run_C3_ambient so the two are directly comparable.
#
# Usage (inside tmux on the GPU server, same as run_next_batch.sh):
#   tmux new -s fark_ambient_batched
#   bash run_ambient_batched.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_ambient_batched
#
# Checkpointed per-week + always passes --resume, so if tmux/SSH drops,
# just rerun this same script -- it picks up where it left off, and any
# already-finished run is skipped entirely.
#
# IMPORTANT: run this on the test/batched-tweet-perception branch (not
# master) -- that's what actually changes the tweet-perception behavior.
#   git checkout test/batched-tweet-perception && git pull

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/src"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "test/batched-tweet-perception" ]; then
  echo "ERROR: on branch '$BRANCH', expected 'test/batched-tweet-perception'." >&2
  echo "Run: git checkout test/batched-tweet-perception && git pull" >&2
  exit 1
fi

export LLM_PROVIDER=local
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

AGENTS="../agents_final_100_seeded.json"
NEWS_CORPUS="../outputs/news/news_corpus_qwen.json"
CONTEXT_CORPUS="../outputs/news/context_corpus_qwen.json"
NETWORK="../outputs/networks/social_network_qwen_noarea.json"
TIMESTEPS=12
CONCURRENCY=32
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
  python driver.py --run-name "$name" --agents "$AGENTS" \
    --timesteps "$TIMESTEPS" --concurrency "$CONCURRENCY" --resume "$@" \
    2>&1 | tee -a "$RUNS_DIR/${name}.batch.log"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $name ====="
}

# C3 + balanced ambient context (real policy news + context articles).
# Matches run_C3_ambient's original recipe exactly.
run run_C3_ambient_batched \
  --condition C3 --network "$NETWORK" --news-corpus "$NEWS_CORPUS" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

# C1 + balanced ambient context (pure social, zero real policy).
# NOTE the flags, same quirk as the original run_C1_ambient: --condition C3
# (NOT C1), --news-corpus omitted. The engine only reads the news schedule
# when policy_on=True (engine.py), and C1 has policy_on=False by definition,
# so plain `--condition C1 --context-corpus ...` would silently deliver ZERO
# context. Omitting --news-corpus here makes build_news_schedule fall into
# its context-only branch (news.py: context_only = context_corpus_path is
# not None and corpus_path is None) -- zero policy text, context articles
# only. The saved run JSON's "condition" field will say "C3" despite
# carrying no policy content -- same documented quirk as the original run,
# not a bug.
run run_C1_ambient_batched \
  --condition C3 --network "$NETWORK" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

echo
echo "Both ambient reruns complete. Outputs in $RUNS_DIR/{run_C3_ambient_batched,run_C1_ambient_batched}.json"
echo "All 4 batched-perception reruns now done: run_C1_Qwen_batched, run_C3_Qwen_batched, run_C1_ambient_batched, run_C3_ambient_batched"
