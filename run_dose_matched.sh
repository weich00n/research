#!/bin/bash
# DOSE-MATCHED policy category comparison — the confound check.
#
# THE PROBLEM THIS EXISTS TO SOLVE
# The headline result (caregiving beats financial on intention) is confounded by
# novelty rate. Financial has 3 instruments and caregiving 5, so over 12 weeks
# each financial instrument repeats ~4 times against caregiving's ~2.4. A
# week-by-week decomposition of the existing runs shows the category gap is
# ABSENT while both categories deliver novelty at the same rate (weeks 1-3:
# diff -0.004, -0.004, +0.025, all n.s.) and opens exactly when those rates
# diverge (week 4: +0.059, p=7.8e-4; week 5: +0.084, p=8.6e-6).
#
# So two explanations currently fit the data equally well:
#   (a) caregiving policy content moves intention more  <- what the paper claims
#   (b) NEW instruments move intention and repeats do not, so the category with
#       more distinct instruments wins regardless of content
#
# This script equalises the schedule shape: caregiving is restricted to three
# instruments so both categories run 3 x 4 exposures. If caregiving still wins,
# (b) is ruled out and the claim is clean. If it does not, the finding is that
# policy VARIETY beats policy REPETITION — still publishable, but a different
# paper, and far better discovered now than in review.
#
# The three caregiving instruments are chosen for real-world evidence coverage,
# not for effect size: Flexible Work Arrangements (Wang & Dong 2024 survey
# experiment, the only experimental benchmark in the policy set), Enhanced
# Paternity Leave (Yeung et al. 2023 behavioural null), and Preschool & Infant
# Care Subsidies. Financial is UNCHANGED, so its arm is directly comparable to
# the existing runs.
#
# 4 runs x 100 agents x 12 weeks, ~20 min each, ~80 min total.
#
# Usage (GPU server, inside tmux):
#   tmux new -s dose_matched
#   bash run_dose_matched.sh
#   # detach: Ctrl-b d ; reattach: tmux attach -t dose_matched

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

CAREGIVING_THREE=("Flexible Work Arrangement Request Guidelines" \
                  "Enhanced Paternity Leave" \
                  "Preschool & Infant Care Subsidies")

# --- preflight -------------------------------------------------------------
echo "Preflight:"
[ -f "$AGENTS" ] || { echo "  ERROR: missing $AGENTS" >&2; exit 1; }
if [ ! -f "$NEWS_CORPUS" ]; then
  echo "  ERROR: missing $NEWS_CORPUS" >&2
  echo "  outputs/ is gitignored, so the corpus does NOT arrive via git pull." >&2
  exit 1
fi
python - <<'PY' || exit 1
import json, sys
p = "../outputs/news/news_corpus_qwen.json"
try:
    d = json.load(open(p))
except Exception as e:
    print(f"  ERROR: {p} is not valid JSON ({type(e).__name__}).", file=sys.stderr)
    print("  This file has been corrupted before by text pasted into it.", file=sys.stderr)
    sys.exit(1)
a = d["articles"] if isinstance(d, dict) else d
if len(a) != 45:
    print(f"  ERROR: corpus has {len(a)} articles, expected 45.", file=sys.stderr)
    sys.exit(1)
print(f"  OK: corpus parses, {len(a)} articles")
PY

PROVIDER="$(python -c "from utils.generate_utils import LLMClient; print(LLMClient().provider)")"
[ "$PROVIDER" = "local" ] || { echo "  ERROR: provider='$PROVIDER', expected 'local'" >&2; exit 1; }
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
  run "run_C2_direct_financial_dose3_s$s"  --policy-category financial
  run "run_C2_direct_caregiving_dose3_s$s" --policy-category caregiving \
      --policy-names "${CAREGIVING_THREE[@]}"
done

echo
echo "Done. Bring back with:"
echo "  scp 'USER@HOST:/data1/USER/research/outputs/runs/run_C2_direct_*_dose3_s*.*' outputs/runs/"
