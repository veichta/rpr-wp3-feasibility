# RPR WP3 feasibility: GAM on Clariden

Feasibility measurements for WP3 (Predict: action-conditioned geometric world modeling) of the
Swiss AI large-grant proposal "Reason, Predict, React". Production stack: the Geometric Action
Model (GAM, `cvlab-kaist/Geometric-Action-Model` @ 58bb91e) on a DA3-Giant backbone
(`ByteDance-Seed/Depth-Anything-3` @ 2c21ea8), NGC PyTorch 25.03 container, GH200 nodes.

Everything lives under one root, `$RPR_WP3_ROOT` (default `/iopsstor/scratch/cscs/veichta/rpr-wp3`).
This folder is a git repo: only `README.md`, `scripts/`, `env/gam.toml` and `.gitignore` are
tracked; `code/`, `assets/`, `env/gam-venv/`, `tools/` and `runs/` are reproduced by the commands
below. A mirror of the tracked files lives in the SplatFactory checkout under
`academic/proposals/cscs-26-large/wp3-feasibility/`.

## Layout

```
README.md            this file
scripts/             setup.sh, env.sh, build_env.sh, check_env.py, gpu_burn.py, *.sbatch
env/gam.toml         container EDF (NGC PyTorch 25.03 + NCCL + DCGM hooks)
env/gam-venv/        Python venv built inside the container (see Create env)
code/gam             GAM source; symlinks Depth-Anything-3 -> ../Depth-Anything-3, checkpoints/ and data/ -> assets/
code/Depth-Anything-3
assets/              track4world_da3.pth (5.5 GB), data/libero_noop/{_stats,_smoke_libero_spatial}, clip-vit-large-patch14/
tools/gssr           GPU Saturation Scorer (gssr-record, gssr-analyze)
runs/                one folder per job: slurm .out, gssr_report/, results/
```

## Setup (login node, needs internet, ~15 min, mostly download)

```bash
export RPR_WP3_ROOT=/iopsstor/scratch/cscs/$USER/rpr-wp3   # optional; this is the default for veichta
bash scripts/setup.sh
```

Clones GAM and DA3 at the pinned commits, creates the three symlinks GAM expects, downloads the
assets from Hugging Face with plain curl (no hf CLI needed), and builds `gssr-record` against the
host DCGM library. Idempotent: re-running skips what exists. The container image path in
`env/gam.toml` must be readable by you (currently the a0231 copy of
`nvcr.io#nvidia/pytorch:25.03-py3`; any copy of that image works).

## Create env (one debug job, ~10 min)

```bash
sbatch scripts/env_build.sbatch
```

Builds `env/gam-venv` on top of the container (`--system-site-packages`, so torch 2.7 comes from
the container) with the pinned GAM requirements minus the MuJoCo/robosuite/LIBERO closed-loop eval
stack, then runs `scripts/check_env.py`. The NGC image pins packages through `PIP_CONSTRAINT`;
`build_env.sh` clears it. Done when `runs/gam-envbuild-<jid>.out` ends with the
`torch ... cuda True gpus 4` and `transformers ... deepspeed ... h5py ...` lines.
`env/gam-venv-freeze.txt` records the exact package set.

## Smoke test (one debug job, ~5 min)

```bash
sbatch scripts/smoke.sbatch
```

60 s of matmuls on all 4 GPUs under `gssr-record`, `gssr-analyze` to PDF, then 3 single-GPU
training steps of GAM on the LIBERO smoke file. Output in `runs/smoke_<jid>/`.

## Scaling runs

```bash
sbatch -N 1 --export=ALL,TAG=ddp scripts/gam_scale.sbatch      # 4 GPUs
sbatch -N 2 --export=ALL,TAG=ddp scripts/gam_scale.sbatch      # 8 GPUs
sbatch -N 3 --export=ALL,TAG=ddp scripts/gam_scale.sbatch      # 12 GPUs
sbatch -N 4 --export=ALL,TAG=ddp scripts/gam_scale.sbatch      # 16 GPUs (debug partition ceiling)
```

Paper config (`chunk8_150k_2node.yaml`: DA3-Giant frozen below block 13, 12-layer predictor,
224 px, 2 views, 8 anchors per sample, 8-step action chunks), global batch fixed at 48 across GPU
counts, 300 steps, torchrun DDP (add `DS=1` for DeepSpeed ZeRO-2). Throughput is the `s/step`
field of the training log; each job also leaves a `gssr_report/` for `gssr-analyze`.

Debug partition limits (a144 works there): max 4 nodes, 90 node-minutes per job, 1 running + 1
queued job per user. GAM stops training and checkpoints when 600 s of SLURM walltime remain, so a
20 min job yields ~6.5 min of steps after the ~3.5 min startup (checkpoint + CLIP load, NCCL init).

## Pull reports to the local mirror

```bash
bash scripts/pull_runs.sh      # from academic/proposals/cscs-26-large/wp3-feasibility/
```

Copies every `runs/<job>/` slurm `.out`, GSSR PDF and raw `gssr_report/` CSVs, and training logs
into `runs/` here (gitignored on both sides).

## GSSR reading of the first scaling series (2026-09-03)

During the training phase every run shows ~70% GPU utilization, ~50-55% SM active, ~20% SM
occupancy, ~15-18% tensor-pipe activity, ~370 W per GPU, and 85-90 GB of the 96 GB used at
micro-batch 3 (so 3 is the largest micro-batch that fits). GSSR's headline "Total Runtime" ratings
are dragged down by the startup phase (checkpoint + CLIP load, NCCL init: ~110-160 s with no GPU
activity): 4 GPUs 52.6% "acceptable" over 11 min, 16 GPUs 33.9% "improve" over 4.8 min. The
"After GPU Util > 50%" column is the fairer one (56% / 35% because the tail is included too).
Consequence for the submission runs: make the training phase long (>= 15 min) so startup is a
small fraction, and quote the "after" column.

## Job log

| job     | what                          | result                                                                                                                         |
| ------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 3276849 | env build                     | FAILED: NGC `PIP_CONSTRAINT` pins regex, transformers 5.5.4 needs newer                                                        |
| 3276857 | env build + gssr burn + smoke | venv OK, gssr burn OK, smoke failed on `--set` syntax (one `--set` per override)                                               |
| 3276947 | smoke                         | gssr burn + PDF OK; training failed: `depth_anything_3` not on sys.path (GAM looks under src/); fixed via PYTHONPATH in env.sh |
