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
git checkout fix/tpb-ratchet-drift
```

## B. Python environment + dependencies

Conda (what actually works reliably — use this):
```bash
conda create -n fark python=3.10 -y
conda activate fark
pip install --upgrade pip
pip install -r requirements.txt            # includes vllm + sentence-transformers
```
venv is the alternative if you don't have conda (`python3 -m venv .venv && source
.venv/bin/activate`), but conda has been the smoother path in practice.

If `pip install vllm` conflicts with the box's CUDA, install the matching torch
first, then vllm. `sentence-transformers` fetches `all-MiniLM-L6-v2` (~80 MB) on
first use.

Every subsequent shell/tmux window in this guide needs `conda activate fark`
(substitute your env name) before running Python — swap in for any `source
.venv/bin/activate` you see below if you went the venv route instead.

## C. Point the HuggingFace cache at big storage (if home is small)
```bash
export HF_HOME=/path/with/space/hf-cache   # e.g. /scratch/$USER/hf-cache
# optional (avoids the rate-limit warning): export HF_TOKEN=hf_xxx
```
Put these in `~/.bashrc` or an `env.sh` you `source`, so every shell has them.

## D. Start Qwen (in tmux so it survives disconnects)

**2-GPU box**: launch only 2 replicas — `GPUS="0 1"` gives ports 8001/8002.
Adjust `GPUS` to however many GPUs you actually have.
```bash
tmux new -s vllm
conda activate fark
   # same as C, inside this tmux
GPUS="0 1" bash serve_qwen.sh              # 2 GPUs -> ports 8001, 8002
```
First launch downloads Qwen2.5-14B (~28 GB) — wait for it. Detach with
**Ctrl-b then d**. Sanity check in another shell:
```bash
curl -s localhost:8001/v1/models
curl -s localhost:8002/v1/models
```

## E. Configure `.env` (repo root, on the server)
```bash
cat > .env <<'EOF'
LLM_PROVIDER=local
LOCAL_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
LOCAL_LLM_URLS=http://localhost:8001/v1,http://localhost:8002/v1
EOF
```
Match the URL list to the ports `serve_qwen.sh` actually launched (2 URLs for a
2-GPU box, not 4).

## F. Run the relevance-mode comparison (new tmux windows, in the same `vllm`
session so everything's in one place — `Ctrl-b c` creates a window, `Ctrl-b w`
lists/switches, `Ctrl-b d` detaches the whole session)

```bash
tmux attach -t vllm
Ctrl-b c                                   # new window
conda activate fark
cd ~/research/src

# 1) the shared frozen baseline (seeds + t=0) is committed in the repo as
#    agents_final_100_seeded.json — do NOT rerun --init-only (that would mint a
#    DIFFERENT baseline and break comparability with the existing runs). Only if
#    the file were ever missing: python driver.py --init-only --relevance-mode hybrid
```

**Ratchet-fix smoke test (current priority — validates the habituation fix in
`prompts.py`/`news.py` before any full run):**
```bash
# window "smoke"
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode hybrid --num-agents 15 --timesteps 12 \
    --run-name c2_smoke_ratchet_fix --output-dir ../outputs/runs/smoke

# readout: "ratchet <construct> monotone non-decreasing / >=4.8 at t=max" lines.
# Old (broken) mechanics for reference: att 100%/56%, norm 92%/67%, pbc 100%/33%.
# Pass = monotone fractions well below ~50% and few agents at the ceiling.
python validation/plot_trajectories.py \
    --runs ../outputs/runs/smoke/c2_smoke_ratchet_fix.json --labels fixed
```

**Pilot (sanity-check at small scale, one window per mode):**
```bash
# window "c2_cosine"
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode cosine --num-agents 20 --timesteps 4 --run-name c2_cosine

# Ctrl-b c -> window "c2_hybrid"
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode hybrid --num-agents 20 --timesteps 4 --run-name c2_hybrid

# Ctrl-b c -> window "c2_llm"
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode llm --num-agents 20 --timesteps 4 --run-name c2_llm
```

**Validation slice (larger n, llm vs hybrid only — see
`outputs/analysis/relevance_mode_decision.md` for why not all 3 modes at
scale):**
```bash
# window "c2_llm_45"
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode llm --num-agents 45 --timesteps 4 --run-name c2_llm_45

# Ctrl-b c -> window "c2_hybrid_45"
python driver.py --condition C2 --agents ../agents_final_100_seeded.json \
    --relevance-mode hybrid --num-agents 45 --timesteps 4 --run-name c2_hybrid_45
```

**Full experiment (once pilots look sane):** `--num-agents 100 --timesteps 12`,
run **hybrid only** per the decision doc, for each of C0-C3. Runs are
checkpointed every week; add `--resume` (same `--run-name`) only to continue a
run that was interrupted partway — don't add it on a fresh launch.

Detach the whole session with **Ctrl-b then d** once everything's launched;
`tmux attach -t vllm` to check back in, `Ctrl-b w` to switch windows.

## G. Compare the results (existing tools; reports land in `outputs/analysis/`)
```bash
cd ~/research/src

# per-memory: how the hybrid run's LLM relevance diverges from cosine
python validation/check_relevance_cosine.py --runs ../outputs/runs/c2_hybrid.json --labels hybrid

# downstream: do the TPB / intention trajectories differ between modes?
# --logs must be passed explicitly -- it does NOT default from --runs and
# silently falls back to stale run_C0*.log paths, producing n/a speed/robustness rows.
python validation/compare_runs.py \
    --runs ../outputs/runs/c2_cosine.json ../outputs/runs/c2_hybrid.json \
    --logs ../outputs/runs/c2_cosine.log ../outputs/runs/c2_hybrid.log \
    --labels cosine hybrid
```

## Notes
- Keep both vLLM and the run in `tmux`; if the box sleeps or SSH drops, the run
  survives — resume with `--resume`.
- Do **not** switch `--relevance-mode` mid-run (resuming a cosine/llm checkpoint in
  hybrid leaves its memories un-shortlistable). Separate fresh runs per mode is fine.
- Multiple driver runs can share the same vLLM endpoints at once (continuous
  batching interleaves requests) — each run will be somewhat slower than
  running solo, but total wall-clock across concurrent runs beats running them
  sequentially.
- Driving a server-side vLLM from your laptop instead: tunnel with
  `ssh -L 8001:localhost:8001 <user>@<host>` (per port), but running the driver
  on the server is simpler.
