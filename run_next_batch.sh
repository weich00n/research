#!/bin/bash
# Run the queued C2-financial/caregiving (policy-only + ambient) and C1/C3
# arms sequentially against ONE vLLM endpoint. Run this INSIDE tmux on the
# GPU server (the local machine has no GPU to serve Qwen).
#
# Usage:
#   tmux new -s fark_batch
#   bash run_next_batch.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_batch
#
# Each driver.py run is checkpointed per-week and this script always passes
# --resume, so if tmux/SSH drops mid-run, just rerun this same script — the
# in-progress run picks up from its last completed week and any already
# FINISHED runs are skipped (see run() below), so it's safe to rerun from
# scratch after a partial batch.
#
# Requires: vLLM already serving Qwen2.5-14B (see serve_qwen.sh) and .env
# resolving to LLM_PROVIDER=local — this script also force-exports that so
# it can never silently fall back to OpenRouter (see preflight check below).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/src"

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

run run_C2_financial \
  --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category financial

run run_C2_financial_ambient \
  --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category financial \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C2_caregiving \
  --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category caregiving

run run_C2_caregiving_ambient \
  --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category caregiving \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

run run_C1_Qwen \
  --condition C1 --network "$NETWORK"

run run_C3_Qwen \
  --condition C3 --network "$NETWORK" --news-corpus "$NEWS_CORPUS"

# --- added 2026-07-21 ---

# Replicate of run_C2_financial with a fresh run-name (same everything else,
# stochastic LLM sampling only) — stability check: is the financial
# specificity finding a stable pattern or this-run's luck?
run run_C2_financial_rep2 \
  --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category financial

# C3 + balanced ambient context — does the same intervention that broke
# C2's ratchet also break C1/C3's echo-chamber norm saturation?
run run_C3_ambient \
  --condition C3 --network "$NETWORK" --news-corpus "$NEWS_CORPUS" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

# Replicate of run_C2_caregiving — pairs with the financial replicate above
# so the stability check covers both policy categories, not just one.
run run_C2_caregiving_rep2 \
  --condition C2 --news-corpus "$NEWS_CORPUS" --policy-category caregiving

# C1 + balanced ambient context — pairs with run_C3_ambient. C1_fixed's
# norm saturation (96%) was even worse than C3's (91%), so this checks
# whether ambient context resolves the pure-social echo chamber too, not
# just the policy+social one.
#
# NOTE the flags: --condition C3 (NOT C1), --news-corpus omitted. The engine
# only ever reads the news schedule when policy_on=True (engine.py:393), and
# C1 has policy_on=False by definition -- so plain `--condition C1
# --context-corpus ...` would silently deliver ZERO context (wasted GPU
# time, looks identical to run_C1_Qwen). To get "social + ambient context,
# no real policy" you need social_on=True (only C1/C3 have it) AND
# policy_on=True (to make the engine read the schedule at all) -- C3 is the
# only condition with both. Omitting --news-corpus then makes
# build_news_schedule fall into its dedicated context_only branch
# (news.py:148: `context_only = context_corpus_path is not None and
# corpus_path is None`) -- zero policy text, context articles only, exactly
# like run_context_only did under --condition C2 in the first batch. The
# saved run JSON's condition field will literally say "C3" despite carrying
# no policy content -- same documented quirk as run_context_only, not a bug.
run run_C1_ambient \
  --condition C3 --network "$NETWORK" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix balanced

echo
echo "All 10 runs complete. Outputs in $RUNS_DIR/{run_C2_financial,run_C2_financial_ambient,run_C2_caregiving,run_C2_caregiving_ambient,run_C1_Qwen,run_C3_Qwen,run_C2_financial_rep2,run_C3_ambient,run_C2_caregiving_rep2,run_C1_ambient}.json"
