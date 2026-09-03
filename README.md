# RPR WP3 feasibility: GAM on Clariden

Feasibility measurements for WP3 (Predict: action-conditioned geometric world modeling) of the
Swiss AI large-grant proposal "Reason, Predict, React". Production stack: the Geometric Action
Model (GAM, `cvlab-kaist/Geometric-Action-Model` @ 58bb91e) on a DA3-Giant backbone
(`ByteDance-Seed/Depth-Anything-3` @ 2c21ea8), NGC PyTorch 25.03 container, GH200 nodes.

Everything lives under one root, `$RPR_WP3_ROOT` (default `/iopsstor/scratch/cscs/veichta/rpr-wp3`).
Git: `git@github.com:veichta/rpr-wp3-feasibility.git`. Only `README.md`, `WP3_feasibility.md`,
`scripts/`, `env/gam.toml` and `.gitignore` are tracked; `code/`, `assets/`, `env/gam-venv/`,
`tools/` and `runs/` are reproduced by the commands below. The working copy that commits and pushes
is the local mirror in the SplatFactory checkout (`academic/proposals/cscs-26-large/wp3-feasibility/`);
the Clariden root receives the tracked files by rsync (Clariden has no GitHub key). Anyone else:
`git clone` on Clariden over https, then follow Setup.

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
counts, `STEPS` steps (default 300; `-t` and `STEPS` set per run), torchrun DDP (add `DS=1` for
DeepSpeed ZeRO-2). Throughput is the `s/step` field of the training log (patched to three
decimals, see `setup.sh`); each job records `gssr_report/` and writes `gssr_<jobid>.pdf` at the
end (`scripts/gssr_pdfs.sbatch` does the same standalone). `env.sh` turns on GAM's data-loader
wait logging (rank 0, every 10 steps) and sets GAM's walltime guard to 90 s (default 600 s: it
stops and checkpoints that long before the SLURM limit). Startup (checkpoint + CLIP load, NCCL
init) takes ~2-3.5 min before the first step.

Debug partition limits (a144 works there): max 4 nodes, 90 node-minutes per job, 1 running + 1
queued job per user. Convention (2026-09-03): every measurement job is 20 min, the longest that fits
all node counts on debug (4 nodes x 22 min cap); the debug docs forbid production-like workloads,
so no hour-long runs there.

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

| job     | what                                                         | result                                                                                                                                                                                                      |
| ------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3276849 | env build                                                    | FAILED: NGC `PIP_CONSTRAINT` pins regex, transformers 5.5.4 needs newer                                                                                                                                     |
| 3276857 | env build + gssr burn + smoke                                | venv OK, gssr burn OK, smoke failed on `--set` syntax (one `--set` per override)                                                                                                                            |
| 3276947 | smoke                                                        | gssr burn + PDF OK; training failed: `depth_anything_3` not on sys.path (GAM looks under src/); fixed via PYTHONPATH in env.sh                                                                              |
| 3277240 | smoke                                                        | model loads (1033.4M trainable params); 0 train samples: the smoke file has ONE demo and `eval_ratio=0.05` reserves it for eval; fixed with `dataset.eval_ratio=0`                                          |
| 3277295 | smoke                                                        | COMPLETED: gssr burn + PDF, 3 single-GPU training steps (first step 10.5 s incl. warmup)                                                                                                                    |
| 3277344 | scale 4 GPU, DDP, 200 steps                                  | COMPLETED 11:03 wall: steps 20-140 at 3.0-3.6 s/step, mean 3.22 = 14.9 samples/s (3.73 per GPU); stopped at step 146 by GAM's 600 s walltime guard                                                          |
| 3277415 | scale 8 GPU (2 nodes), 150 steps                             | COMPLETED 7:03: steps 20-150 at 1.2-2.0 s/step, mean 1.64 = 29.2 samples/s (3.66 per GPU), 98% of linear                                                                                                    |
| 3277469 | scale 12 GPU (3 nodes), 150 steps                            | training COMPLETED 5:48: steps 20-150 at 1.0-1.5 s/step, mean 1.26 = 38.2 samples/s (3.18 per GPU), 85%; data-loader wait 0.00 s; job FAILED only in the PDF stage (srun flags swallowed by numactl, fixed) |
| 3277556 | scale 16 GPU (4 nodes), 150 steps                            | training COMPLETED 4:49: steps 20-150 at 0.6-1.0 s/step, mean 0.72 = 66.6 samples/s (4.16 per GPU), 112% = one-decimal rounding; data-loader wait 0.00 s; same PDF-stage failure                            |
| 3277648 | gssr PDFs for the four scaling runs                          | COMPLETED, PDFs in runs/ddp_*gpu_*/                                                                                                                                                                         |
| 3277709 | scale 4 GPU rerun, data-wait logging, 150 steps              | COMPLETED 10:36: 3.2-3.3 s/step, wait 0.00 s at every logged batch -> baseline is compute-bound; PDF in-job                                                                                                 |
| 3277723 | scale 8 GPU rerun, 3-decimal s/step, 150 steps               | COMPLETED 7:25: steps 20-150 mean 1.540 s/step (1.06-1.90, std ~0.2) = 31.2 samples/s (3.90 per GPU), 104% of linear; wait 0.00 s; multi-node windows jitter ±30% while the single-node run is steady       |
| 3277872 | long 4 GPU, 1 node x 60 min, 1000 steps (submission series) | COMPLETED 57:50, all 1000 steps: mean 3.333 s/step (std 0.287, 2.70-4.33) = 14.40 samples/s (3.60 per GPU); GSSR over the whole job: GPU util 66.3% acceptable, FP util 0.17 acceptable, 373 W, GPU start 97% good |
| 3277945 | long 8 GPU, 2 nodes x 40 min, 1200 steps (submission series) | COMPLETED 38:08, all 1200 steps: mean 1.657 s/step (std 0.195, 1.14-2.23) = 28.97 samples/s (3.62 per GPU), 100.6% of linear vs the long 4-GPU run |
| 3278593 | long 12 GPU, 3 nodes x 28 min, 1100 steps (submission series) | 1071 steps, then guard + 12 GB checkpoint + PDF overran the limit (TIMEOUT after the PDF was written): mean 1.282 s/step (std 0.116, n=105 windows) = 37.4 samples/s (3.12 per GPU), 86.6% of linear (micro-batch 2); one 121 s data-loader stall at a virtual-epoch boundary (rank 0, step 501) excluded as an outlier window (13.5 s/step) |
| 3279116 | long 16 GPU, 4 nodes x 22 min, 1300 steps (submission series) | TIMEOUT at 22:11 after 1220 steps (SLURM killed it before the guard/PDF stage): mean 0.868 s/step (std 0.144, no outliers, n=121) = 55.3 samples/s (3.46 per GPU), 96.0% of linear; data-loader wait 0.00 s at all 122 logged batches; PDF still to be generated from gssr_report/ with scripts/gssr_pdfs.sbatch |
| 3279461 | ZeRO-2 (DeepSpeed) 4 GPU, micro 3 x accum 4, 150 micro-steps: launcher equivalence check | COMPLETED: GAM counts micro-batches as steps on the DeepSpeed path (lr schedule confirms 4 micro-steps per optimizer step); mean 0.992 s per 12-sample micro-step = 12.1 samples/s (3.02 per GPU) = 84% of DDP; ZeRO-2 is slower for this model (no memory need, extra reduce-scatter/all-gather), so DDP is the production launcher; its PDF stage also produced the missing 16-GPU PDF |
| 3279908 | full fine-tune (all DA3 blocks, 1401.9M trainable) + ZeRO-2, micro 1 x accum 12, 4 GPU, 400 micro-steps, 5:27 wall | 0.518 s per 4-sample micro-step = 7.7 samples/s (1.93 per GPU, 54% of the frozen-encoder DDP baseline); only ~29 GB used per GPU; GSSR poor: GPU util 43-46% improve, FP util 0.03-0.04 poor, 190 W improve -> micro-batch 1 starves the GPU |
| 3280114 | full fine-tune + ZeRO-2, micro 1 x accum 12, 448 px (1024 patches/view, 16,384 visual tokens/sample), 4 GPU, 400 micro-steps, ~8 min | 1.005 s per 4-sample micro-step (std 0.091) = 3.98 samples/s (1.00 per GPU = 52% of the 224 px full-FT run for 4x the tokens); 70.8 GB used per GPU; GSSR: GPU util 75-79% good, FP util 0.11-0.12 improve, 300 W acceptable, GPU start 83% good |
| 3280203 | full fine-tune + ZeRO-2, micro 1 x accum 12, 518 px native DA3 (1369 patches/view, 21,904 visual tokens/sample), 4 GPU, 250 micro-steps, 10 min | 1.384 s per 4-sample micro-step (std 0.150) = 2.89 samples/s (0.72 per GPU, 73% of 448 px for 1.34x tokens); 89.6 GB used per GPU (micro 1 is the ceiling); GSSR: GPU util 77-80% good, FP util 0.12-0.13 improve/acceptable, 300 W acceptable, GPU start 84% good |
| 3280361 | contract probe 224 px, attempt 1 | FAILED at config validation: `len(dataset.camera_keys)` must equal `n_views` (view repetition happens later in the loader) |
| 3280364 | contract probe 224 px, attempt 2 (3 explicit camera keys) | FAILED at config validation: `da3_finetune.n_action_steps` must equal `future_steps + 1` (16) |
| 3280377 | contract probe 224 px, attempt 3 (n_action_steps 16) | FAILED: 0 samples; 16 anchors at 8-frame chunk need >= 128 frames, the smoke demo has 98 (uniform sampling does not rescue a demo shorter than the window) |
| 3280385 | contract probe 224 px: 3 views (external duplicated), 16 anchors at 4-frame chunk (stand-in for the 98-frame demo), full fine-tune, lambda_point=1, deep gradient checkpointing, ZeRO-2 micro 1 x accum 12, 200 micro-steps, ~9 min | 1.484 s per 4-sample micro-step (std 0.092) = 2.70 samples/s (0.67 per GPU) for 12,288 visual tokens/sample; only 27 GB used per GPU (checkpointing) -> micro 3 fits; GSSR: GPU util 58-61% acceptable, FP util 0.07-0.08 improve, 240 W; point loss enabled (lambda_point=1.0 in the loss line) but the point term logs 0.0 |
| 3280626 | contract probe 518 px with accum 12 | cancelled after 32 s (replaced by the no-accumulation run) |
| 3280628 | contract probe 518 px: 3 views (external duplicated), 16 anchors at 4-frame chunk, full fine-tune, `depth_loss_type=da3_full` with point + ray losses (live: point 0.37, ray 1.5), deep gradient checkpointing, ZeRO-2 micro 1, NO accumulation (global batch 4), 46 steps in 10 min | 9.55 s per step (1 sample per GPU, std 1.16) = 0.105 samples/s/GPU = 6.9k tokens/s/GPU for 65,712 visual tokens/sample; 80 GB used per GPU (86 max); GSSR: GPU util 82-85% good, FP util 0.16-0.17 acceptable, 310 W acceptable, GPU start 87% good -> best scores so far |
| 3280912 | contract 518 px as 3280628 but gradient accumulation 4 (global batch 16) | 9.39 s per micro-step (std 1.15) = 0.106 samples/s/GPU, 81 GB used; GSSR: GPU util 83-86% good, FP util 0.16-0.17 acceptable, 305 W -> identical to the no-accumulation run: the optimizer/ZeRO step is not the FP-util limiter |
| 3281016 | contract 518 px, accum 4, + `depth_decode_chunk_size=16` + `training.compile=true` | 9.21 s per micro-step (std 1.07) = 0.109 samples/s/GPU (+3% vs 9.39), 83 GB; GSSR: GPU util 86-88% good, FP util 0.18-0.19 acceptable (from 0.16-0.17), 321 W; compile adds ~1 min startup |
