# Running FARK-Sim on the GPU server (Qwen via vLLM)

Setup for serving **Qwen2.5-14B-Instruct** with vLLM and running the simulation /
the cosine-vs-hybrid relevance comparison.

These steps assume a **GPU box you SSH into directly and run inside `tmux`** (matches
`serve_qwen.sh`). If your cluster is different, adjust as noted.

## 0. Confirm these about your cluster first
1. **Login/host** — the `ssh` target (`<user>@<host>`).
2. **GPU access** — direct SSH to the GPU box (assumed here) **or** SLURM
   (`srun --gres=gpu:4 --pty bash`)? If SLURM, wrap Parts D–F in an `sbatch` job.
3. **Internet on the GPU node** — needed for `pip install` and the ~28 GB Qwen
   download. If the compute node is offline, pip-install and pre-download the model
   from a login node first.
4. **CUDA / modules** — do you `module load cuda/python`? Is there big storage
   (e.g. `/scratch/$USER`) for the HuggingFace cache? (Home dirs rarely fit 28 GB.)

## A. Connect and get the code
```bash
ssh <user>@<host>
cd ~/research                # or: git clone https://github.com/weich00n/research.git && cd research
git fetch origin
git checkout feature/baseline-hybrid-relevance
```

## B. Python environment + dependencies
```bash
# module load python/3.10 cuda/12.1        # only if your cluster uses modules
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt            # includes vllm + sentence-transformers
```
If `pip install vllm` conflicts with the box's CUDA, install the matching torch
first, then vllm. `sentence-transformers` fetches `all-MiniLM-L6-v2` (~80 MB) on
first use.

## C. Point the HuggingFace cache at big storage (if home is small)
```bash
export HF_HOME=/path/with/space/hf-cache   # e.g. /scratch/$USER/hf-cache
# optional (avoids the rate-limit warning): export HF_TOKEN=hf_xxx
```
Put these in `~/.bashrc` or an `env.sh` you `source`, so every shell has them.

## D. Start Qwen (in tmux so it survives disconnects)
```bash
tmux new -s vllm
source .venv/bin/activate
export HF_HOME=/path/with/space/hf-cache   # same as C, inside this tmux
bash serve_qwen.sh                         # GPUs 0-3 -> ports 8001..8004
#   GPUS="0 1 3" bash serve_qwen.sh        # skip a busy GPU
```
First launch downloads Qwen2.5-14B (~28 GB) — wait for it. Detach with
**Ctrl-b then d**. Sanity check in another shell:
```bash
curl -s localhost:8001/v1/models
```

## E. Configure `.env` (repo root, on the server)
```bash
cat > .env <<'EOF'
LLM_PROVIDER=local
LOCAL_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
LOCAL_LLM_URLS=http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1
EOF
```
Match the URL list to the ports `serve_qwen.sh` actually launched.

## F. Run the cosine-vs-hybrid comparison (second tmux window, venv activated)
```bash
cd ~/research/src

# 1) freeze ONE shared baseline (seeds + t=0) — run once, both modes load it
python driver.py --init-only --relevance-mode hybrid

# 2) same seeded agents, two relevance modes (start small to sanity-check)
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode cosine --num-agents 20 --timesteps 4 --run-name c2_cosine
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode hybrid --num-agents 20 --timesteps 4 --run-name c2_hybrid
```
Scale up to `--num-agents 100 --timesteps 12` once the small slice looks sane. Runs
are checkpointed every week; use `--resume` (same `--run-name`) to continue an
interrupted run.

## G. Compare the results (existing tools; reports land in `outputs/analysis/`)
```bash
cd ~/research/src

# per-memory: how the hybrid run's LLM relevance diverges from cosine
python validation/check_relevance_cosine.py --runs ../outputs/runs/c2_hybrid.json --labels hybrid

# downstream: do the TPB / intention trajectories differ between modes?
python validation/compare_runs.py \
    --runs ../outputs/runs/c2_cosine.json ../outputs/runs/c2_hybrid.json \
    --labels cosine hybrid
```

## Notes
- Keep both vLLM and the run in `tmux`; if the box sleeps or SSH drops, the run
  survives — resume with `--resume`.
- Do **not** switch `--relevance-mode` mid-run (resuming a cosine/llm checkpoint in
  hybrid leaves its memories un-shortlistable). Separate fresh runs per mode is fine.
- Driving a server-side vLLM from your laptop instead: tunnel with
  `ssh -L 8001:localhost:8001 <user>@<host>` (per port), but running the driver
  on the server is simpler.
