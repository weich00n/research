#!/bin/bash
# Serve Qwen2.5-14B for the fertility ABM on a multi-GPU box (e.g. 4x L40S).
#
# Default: launch one independent vLLM replica per GPU (data-parallel), each on
# its own port 8001..800N. Each L40S (46GB) holds a full bf16 copy (~28GB), so no
# tensor-parallel / NVLink is needed. The driver round-robins across the ports via
# LOCAL_LLM_URLS, and vLLM's continuous batching turns the concurrent agent
# requests into GPU throughput.
#
# Usage:
#   bash serve_qwen.sh            # 4 GPUs (0,1,2,3) -> ports 8001..8004
#   GPUS="0 1 3" bash serve_qwen.sh   # skip a contended GPU (e.g. GPU 2 in use)
#
# Then set in .env (one URL per launched port):
#   LLM_PROVIDER=local
#   LOCAL_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
#   LOCAL_LLM_URLS=http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1
#
# Run inside tmux/screen so the servers survive your SSH session. Ctrl-C stops all.
#
# Single-endpoint alternative (newer vLLM with online data-parallel support):
#   vllm serve Qwen/Qwen2.5-14B-Instruct --data-parallel-size 4 --max-model-len 8192 --port 8000
#   ...and set LOCAL_LLM_URL=http://localhost:8000/v1 instead of LOCAL_LLM_URLS.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BASE_PORT="${BASE_PORT:-8000}"   # first replica = BASE_PORT+1
GPUS="${GPUS:-0 1 2 3}"

echo "Serving $MODEL  (max_model_len=$MAX_MODEL_LEN)  on GPUs: $GPUS"

i=0
for g in $GPUS; do
  i=$((i + 1))
  port=$((BASE_PORT + i))
  echo "  -> GPU $g  ->  http://localhost:$port/v1"
  CUDA_VISIBLE_DEVICES="$g" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size 1 --port "$port" &
done

echo "Launched $i replica(s). Waiting for them to come up..."
echo "Sanity check (in another shell):  curl -s localhost:$((BASE_PORT + 1))/v1/models"
wait
