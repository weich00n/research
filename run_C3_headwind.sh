#!/bin/bash
# Fill the one missing TPB arm: C3_context_headwind (social + pure-negative
# ambient context, no real policy news). Direct and Recsys already have this
# arm (run_C3_direct_context_headwind, run_C3_recsys_context_headwind); TPB
# never had it run -- compare_three_methodologies.py's MATCHED_ARMS carries
# it as `None` for the tpb column and reports it as genuinely missing.
#
# Same documented trick as run_C1_ambient / run_C3_ambient (see
# run_next_batch.sh): --condition C3 gives social_on=True AND policy_on=True
# (needed so the engine reads the news schedule at all), but --news-corpus is
# deliberately OMITTED. build_news_schedule then falls into its context-only
# branch (news.py: context_only = context_corpus_path is not None and
# corpus_path is None) -- zero policy text, negative-valence context articles
# only. The saved run JSON's "condition" field will say "C3" despite no
# policy content -- same documented quirk as run_C1_ambient, not a bug.
#
# Usage (inside tmux on the GPU server -- LLM calls need the vLLM endpoint,
# see serve_qwen.sh; this repo has no GPU locally):
#   tmux new -s fark_c3_headwind
#   bash run_C3_headwind.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t fark_c3_headwind
#
# Checkpointed per-week + always passes --resume, so if tmux/SSH drops, just
# rerun this same script -- it picks up from the last completed week.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/src"

export LLM_PROVIDER=local
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

AGENTS="../agents_final_100_seeded.json"
CONTEXT_CORPUS="../outputs/news/context_corpus_qwen.json"
NETWORK="../outputs/networks/social_network_qwen_noarea.json"
TIMESTEPS=12
CONCURRENCY=32
RUNS_DIR="../outputs/runs"
RUN_NAME="run_C3_headwind"

# --- preflight: fail loudly instead of silently hitting OpenRouter ---
echo "Preflight: checking LLM provider resolves to 'local'..."
PROVIDER="$(python -c "from utils.generate_utils import LLMClient; print(LLMClient().provider)")"
if [ "$PROVIDER" != "local" ]; then
  echo "ERROR: LLM provider resolved to '$PROVIDER', expected 'local'." >&2
  echo "Check .env / LOCAL_LLM_URL(S) and that vLLM (serve_qwen.sh) is running." >&2
  exit 1
fi
echo "  OK: provider=local model=$LOCAL_LLM_MODEL"

RUN_JSON="$RUNS_DIR/${RUN_NAME}.json"
if [ -f "$RUN_JSON" ]; then
  DONE_WEEK="$(python -c "import json; print(json.load(open('$RUN_JSON'))['current_timestep'])" 2>/dev/null || echo -1)"
  if [ "$DONE_WEEK" = "$TIMESTEPS" ]; then
    echo "===== $RUN_NAME already complete (week $DONE_WEEK/$TIMESTEPS) — nothing to do ====="
    exit 0
  fi
fi

echo
echo "===== $(date '+%Y-%m-%d %H:%M:%S')  starting $RUN_NAME ====="
python driver.py --run-name "$RUN_NAME" --agents "$AGENTS" \
  --condition C3 --network "$NETWORK" \
  --context-corpus "$CONTEXT_CORPUS" --context-mix negative \
  --timesteps "$TIMESTEPS" --concurrency "$CONCURRENCY" --resume \
  2>&1 | tee -a "$RUNS_DIR/${RUN_NAME}.batch.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S')  finished $RUN_NAME ====="

echo
echo "Done. Output: $RUN_JSON"
echo "Next: in src/validation/compare_three_methodologies.py, change the"
echo "  (\"C3_context_headwind\", None, ...) row's None to \"$RUN_NAME\","
echo "  then rerun the comparison script to fill in the last missing arm."
