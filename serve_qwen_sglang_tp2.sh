#!/bin/bash
# Serve Qwen2.5-14B via SGLang on a 4x24GB box with NO NVLink (e.g. 4x RTX
# A5000), where a single GPU can't hold a full bf16 replica (needs ~28GB).
#
# Splits the model across GPU PAIRS with tensor-parallel-size 2 instead of
# replicating a full copy per GPU (contrast with serve_qwen.sh, written for
# 4x46GB L40S where each GPU holds its own full replica). Two independent
# TP=2 engines run concurrently (GPUs 0+1 -> port 8101, GPUs 2+3 -> port
# 8102); the driver round-robins across both via LOCAL_LLM_URLS, same as the
# L40S setup.
#
# NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 are REQUIRED here, not optional
# tuning: on this topology (`nvidia-smi topo -m` shows all GPU pairs as
# NODE -- PCIe through the host bridge, no NV# / NVLink), NCCL's P2P
# self-test during init hangs indefinitely (100% GPU util, memory stuck at
# ~400MiB, no error, no timeout -- silent deadlock) unless P2P and shared-
# memory transport are disabled up front, forcing it through host-staged
# transport instead. Confirmed by direct reproduction: with these unset,
# `Init torch distributed` never returns; with them set, weight loading
# proceeds within seconds of the NCCL handshake completing.
#
# ATTENTION_BACKEND=triton (the default here) instead of SGLang's default
# `flashinfer`: on this box, flashinfer's ragged-prefill kernel
# (wrapper_ragged.begin_forward) crashed under real traffic with
# `torch.AcceleratorError: CUDA error: unspecified launch failure` --
# consistent with a flashinfer/CUDA-13.2 compatibility issue on this driver,
# not a config mistake. Triton is the documented fallback and has run stable
# in testing since; if it ALSO crashes, that points to something more
# fundamental (bad GPU / driver / CUDA install) and is worth escalating
# rather than trying yet another backend blind.
#
# Usage:
#   bash serve_qwen_sglang_tp2.sh
#   MODEL=... CONTEXT_LEN=4096 bash serve_qwen_sglang_tp2.sh   # override
#   ATTENTION_BACKEND=flashinfer bash serve_qwen_sglang_tp2.sh # only if you
#                                                               # want to
#                                                               # retest the
#                                                               # crashing
#                                                               # default
#
# Run inside tmux so the servers survive SSH/VS Code disconnects:
#   tmux new -s fark_sglang
#   bash serve_qwen_sglang_tp2.sh
#   Ctrl-b d to detach; tmux attach -t fark_sglang to reattach.
#
# Then in .env (repo root):
#   LLM_PROVIDER=local
#   LOCAL_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
#   LOCAL_LLM_URLS=http://localhost:8101/v1,http://localhost:8102/v1
#
# If your home directory has a tight quota, redirect the ~28GB HuggingFace
# model cache to bulk storage BEFORE first launch (first launch downloads):
#   mkdir -p /data1/<you>/hf_cache && export HF_HOME=/data1/<you>/hf_cache

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
GPU_PAIR_A="${GPU_PAIR_A:-0,1}"
GPU_PAIR_B="${GPU_PAIR_B:-2,3}"
PORT_A="${PORT_A:-8101}"
PORT_B="${PORT_B:-8102}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"

echo "Serving $MODEL (tp=2, context_length=$CONTEXT_LEN, attention_backend=$ATTENTION_BACKEND) on GPU pairs $GPU_PAIR_A -> :$PORT_A, $GPU_PAIR_B -> :$PORT_B"
echo "NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 forced (see script header -- required on non-NVLink boxes)"

CUDA_VISIBLE_DEVICES="$GPU_PAIR_A" NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
  python -m sglang.launch_server --model-path "$MODEL" --tp 2 \
    --attention-backend "$ATTENTION_BACKEND" \
    --context-length "$CONTEXT_LEN" --port "$PORT_A" &

CUDA_VISIBLE_DEVICES="$GPU_PAIR_B" NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
  python -m sglang.launch_server --model-path "$MODEL" --tp 2 \
    --attention-backend "$ATTENTION_BACKEND" \
    --context-length "$CONTEXT_LEN" --port "$PORT_B" &

echo "Launched 2 TP=2 engines. Waiting for them to come up..."
echo "Sanity check (in another shell): curl -s localhost:$PORT_A/v1/models"
wait
