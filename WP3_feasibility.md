# WP3 Predict: feasibility review draft

Draft text for the Swiss AI call-4 Feasibility Review template, WP3 (action-conditioned geometric
world modeling). Sections follow the template order. Numbers marked TODO come from the long
submission runs (2026-09-03 afternoon) or from decisions still open with Philipp/Marc.

## Resource Justification and Timeline (WP3 rows)

Workload = GAM training (the production stack of WP3: DA3-Giant geometric foundation model split
at block 13, 12-layer causal future predictor, action head; 1.4B parameters, 983M trainable in the
paper config, 1033M with the depth head unfrozen as in the LIBERO config).

| Work package [deps]                                                       | Target scale (#GPUs/job) | #runs, #concurrent | Data target                             | Benchmarked? / diff to prod / data ready?                                                                           | Avg throughput [strong-scaling eff.] | Run duration, WP duration | GPUh                  |
| ------------------------------------------------------------------------- | ------------------------ | ------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------- | --------------------- |
| WP3.1 multi-horizon future-geometry prediction [WP1 data, WP2 checkpoint] | 64 (TODO: 64 or 128)     | TODO               | TODO samples (= trajectories x windows) | yes (4-16 GPU) / prod uses 3 views + longer horizon (TODO measure) / LIBERO+OXE public, WP1 conversion for the rest | TODO samples/s [TODO %]              | TODO                      | 0.22 G = 1.01M target |
| WP3.2 counterfactual prediction + uncertainty [3.1]                       | TODO                     | TODO               | TODO                                    | same stack, K candidate action chunks per sample (TODO measure K=4)                                                 | TODO                                 | TODO                      | TODO                  |
| WP3.3 predictive supervision for planning [3.1, 3.2]                      | TODO                     | TODO               | TODO rollouts                           | inference workload, same container (TODO measure)                                                                   | TODO                                 | TODO                      | TODO                  |

Budget formula per job: GPUh = samples_target / (samples_per_s_per_GPU x 3600) / eff + 5 % restarts.

## Application Efficiency on Alps (WP3.1 workload)

Workload: GAM training, `configs/training/libero_unified/gam/chunk8_150k_2node.yaml` (paper
config): 224 px, 2 views (external + wrist), 8 RGB anchors per sample (future_steps 7, observed
history H sampled in 1..7), 8-step action chunks, DA3-Giant blocks 0-12 frozen encoder, blocks 13-39
fine-tuned on the predicted sequence, 12-layer predictor (d=1024, 16 heads, flex_attention
block-causal mask), CLIP ViT-L/14 text conditioning, bf16 autocast, AdamW (lr 5e-5 backbone, 10x
head, 0.2x predictor), depth decode loss on embedded GT depth.

Parallel setup: data parallelism only (torchrun DDP, NCCL over the aws-ofi hook), one process per
GPU, gradient accumulation to hold the global batch at 48 samples; DDP skips gradient sync on
non-boundary micro-steps. Minimal setup that fits: 1 GPU at micro-batch 3 (85-90 GB of 96 GB), so
micro-batch 3 is the largest that fits and 4 GPUs (one node) is the baseline.

Tokens per sample (DA3 patch 14 at 224 px = 256 visual tokens per view):
8 anchors x 2 views x 256 = 4,096 visual tokens through the encoder, plus 77 language tokens and
the action/proprio tokens through the predictor, plus the predicted future frame through blocks 13-39.

### Strong scaling, first series (150-200 steps, one-decimal s/step)

| #GPUs | nodes | micro x accum | s/step (mean, steps 20+) | samples/s | samples/s/GPU | efficiency       | SLURM job | log                    |
| ----- | ----- | ------------- | ------------------------ | --------- | ------------- | ---------------- | --------- | ---------------------- |
| 4     | 1     | 3 x 4         | 3.22                     | 14.9      | 3.73          | 100 %            | 3277344   | runs/ddp_4gpu_3277344  |
| 8     | 2     | 3 x 2         | 1.64                     | 29.2      | 3.66          | 98 %             | 3277415   | runs/ddp_8gpu_3277415  |
| 12    | 3     | 2 x 2         | 1.26                     | 38.2      | 3.18          | 85 %             | 3277469   | runs/ddp_12gpu_3277469 |
| 16    | 4     | 3 x 1         | 0.72                     | 66.6      | 4.16          | 112 % (rounding) | 3277556   | runs/ddp_16gpu_3277556 |

Data-loader wait logged as 0.00 s at every sampled batch on 4, 12 and 16 GPUs (jobs 3277709,
3277469, 3277556): the input side is not limiting. The 12-GPU point runs micro-batch 2 (48 / 12),
which costs efficiency; production global batches are multiples of 3 x #GPUs.

### Strong scaling, submission series (>= 15 min training per point, three-decimal s/step)

| #GPUs | nodes | micro x accum | s/step                                                                              | samples/s | efficiency | SLURM job | GSSR PDF                                                               |
| ----- | ----- | ------------- | ----------------------------------------------------------------------------------- | --------- | ---------- | --------- | ---------------------------------------------------------------------- |
| 4     | 1     | 3 x 4         | 3.333 (std 0.287, 1000 steps)                                                       | 14.40     | 100 %      | 3277872   | runs/long_4gpu_3277872/gssr_3277872.pdf: GPU util 66 %, FP 0.17, 373 W |
| 8     | 2     | 3 x 2         | 1.657 (std 0.195, 1200 steps)                                                       | 28.97     | 101 %      | 3277945   | runs/long_8gpu_3277945/gssr_3277945.pdf                                |
| 12    | 3     | 2 x 2         | 1.282 (std 0.116, 1071 steps; one 121 s loader stall at an epoch boundary excluded) | 37.4      | 87 %       | 3278593   | runs/long_12gpu_3278593/gssr_3278593.pdf                               |
| 16    | 4     | 3 x 1         | 0.868 (std 0.144, 1220 steps)                                                       | 55.3      | 96 %       | 3279116   | runs/long_16gpu_3279116/gssr_3279116.pdf                               |

Launcher check: DeepSpeed ZeRO-2 (job 3279461, same micro 3 x accum 4 on 4 GPUs) reaches 12.1 samples/s vs 14.4 for DDP (84 %); ZeRO-2 only adds sharding traffic for a model whose optimizer state already fits, so DDP is the production launcher.

Beyond 16 GPUs (debug partition ceiling) requires the `normal` partition: TODO 32/64 GPU points
if the queue allows before submission; otherwise state the 16-GPU measurement and the DDP
communication argument (1B trainable params, 4 GB bf16 gradient all-reduce per step, overlapped
with backward).

GSSR (all runs, training phase): ~70 % GPU utilization, ~55 % SM active, ~20 % SM occupancy,
~15-18 % tensor-pipe activity, ~370 W per GPU. Reports: `runs/<run>/gssr_<jobid>.pdf`, raw
DCGM CSVs in `runs/<run>/gssr_report/`. Logs and job scripts: TODO copy to
`/capstor/store/cscs/swissai/a144/Logs/feasibility_review_rpr/WP3/` (csstaff-readable).

### Production-shape probes (4 GPUs, 2026-09-03 afternoon)

| setting                                                                                                           | visual tokens/sample | samples/s/GPU | GB/GPU | GPU Util           | FP Util                      | job     |
| ----------------------------------------------------------------------------------------------------------------- | -------------------- | ------------- | ------ | ------------------ | ---------------------------- | ------- |
| paper config, frozen encoder, DDP micro 3                                                                         | 4,096                | 3.60          | 90     | 66 % acceptable    | 0.17 acceptable              | 3277872 |
| full fine-tune, ZeRO-2 micro 1, 224                                                                               | 4,096                | 1.93          | 29     | 43 % improve       | 0.03 poor                    | 3279908 |
| full fine-tune, ZeRO-2 micro 1, 448                                                                               | 16,384               | 1.00          | 71     | 75-79 % good       | 0.11-0.12 improve            | 3280114 |
| full fine-tune, ZeRO-2 micro 1, 518                                                                               | 21,904               | 0.72          | 90     | 77-80 % good       | 0.12-0.13 improve/acceptable | 3280203 |
| contract 224: 3 views x 16 anchors, full FT, da3_style loss, deep ckpt, micro 1                                   | 12,288               | 0.67          | 27     | 58-61 % acceptable | 0.07-0.08 improve            | 3280385 |
| contract 518: 3 views x 16 anchors, full FT, da3_full loss (depth+ray+point), deep ckpt, micro 1, no accumulation | 65,712               | 0.105         | 80     | 82-85 % good       | 0.16-0.17 acceptable         | 3280628 |
| contract 518, accumulation 4                                                                                      | 65,712               | 0.106         | 81     | 83-86 % good       | 0.16-0.17 acceptable         | 3280912 |
| contract 518, accumulation 4, batched depth decode (16) + torch.compile                                           | 65,712               | 0.109         | 83     | 86-88 % good       | 0.18-0.19 acceptable         | 3281016 |
| contract 518 with H fixed to 15 on every rank (no per-rank load imbalance)                                        | 65,712               | 0.085         | 89     | 82-85 % good       | 0.34-0.35 good               | 3281372 |
| **contract 448** (pi0.7 operating point; native for 480p sources), H=15, same otherwise                           | 49,152               | 0.140         | 70     | 85-88 % good       | 0.37-0.39 good               | 3282051 |

Profiling (torch.profiler on rank 0, job 3281286) showed that with GAM's per-rank sampling of the
history length H (1..15) the ranks do unequal work and 49 % of GPU time is spent waiting in NCCL
all-reduce kernels (median 214 ms per bucket vs 45 ms once balanced); GPU Util counts that spin as
busy, FP Util does not. With H equal on all ranks (job 3281372) compute occupies 98 % of the window
and FP Util reaches 0.35. Patch `patches/gam_sync_H.py` keeps GAM's H curriculum but draws the same
H on every rank per step. The 448 px row is the WP3.1 candidate (pi0.7 runs at 448 x 448 with 3-4 cameras; 448 is native for the 480p raw sources and needs no upsampling argument): native DA3 resolution, three views, 16 horizons, geometry
losses on, the whole model trainable. Budget at this shape: 1 GPU-h = 378 samples, so 400k GPU-h
= 151M samples (~1.5 epochs over ~100M windows) and 1.0M GPU-h = 378M samples (~3.8 epochs).
Probe stand-ins (compute-neutral): third view = duplicated external camera, anchor spacing 4 frames
instead of 8 because the 98-frame smoke demo cannot host 16 anchors at 8.

### Strong scaling, production contract 448 (submission ladder, 2026-09-03 evening)

Global batch 48 held fixed (micro 1 x accumulation 12/6/4/3), 120 optimizer steps per point,
mean over steps >= 20, ZeRO-2, `configs/wp31_contract_448.yaml`, cached single-demo input.

| #GPUs | nodes | accum | s/sample (std)  | samples/s/GPU | samples/s | efficiency | SLURM job | GSSR (`runs/ladder_<n>gpu_<job>/gssr_<job>.pdf`)                  |
| ----- | ----- | ----- | --------------- | ------------- | --------- | ---------- | --------- | ----------------------------------------------------------------- |
| 4     | 1     | 12    | 7.124 (0.014)   | 0.140         | 0.56      | 100 %      | 3283278   | GPU start 93 %, GPU util 92-93 %, FP util 0.41-0.42 (all good), 469 W |
| 8     | 2     | 6     | 7.140 (0.015)   | 0.140         | 1.12      | 99.8 %     | 3283391   | GPU start 92 %, GPU util 91-92 %, FP util 0.41 (all good), 466 W  |
| 12    | 3     | 4     | 7.214 (0.032)   | 0.139         | 1.66      | 98.7 %     | 3283394   | GPU start 93 %, GPU util 91-92 %, FP util 0.40 (all good), 468 W  |
| 16    | 4     | 3     | 7.231 (0.021)   | 0.138         | 2.21      | 98.5 %     | 3283421   | GPU start 93 %, GPU util 91-92 %, FP util 0.40 (all good), 465 W  |

Visual tokens per sample 49,152 -> 6.9k visual tokens/s/GPU at every point. Accumulation only holds
the global batch at 48 for the strong-scaling comparison; at the target scale (64 GPUs, micro 1)
the natural global batch is 64 with no accumulation, and the 16-GPU point (accumulation 3) already
shows the per-step gradient communication of that layout costs under 2 %.

### Inference throughput (WP3.3 proxy, 4 GPUs, job 3283544)

Forward + loss pass of the same 448 contract (`--eval-only`, batch 1, 100 samples per GPU, 400
hard-linked copies of the smoke demo): 197 s active window per GPU (1 s DCGM samples) =
1.97 s/sample = 0.51 samples/s/GPU = 2.03 samples/s per node, 3.6x the training rate
(7.12 s/sample). Peak memory 30.7 GB/GPU (room for batch 3). GSSR over the whole run: GPU util
67-73 % acceptable (the 85 s start-up is inside the window), FP util 0.31-0.34 good (0.45 over
the busy ticks), 383 W; PDF `runs/infer_4gpu_3283544/gssr_3283544.pdf`. WP3.3 rollouts are
this pass repeated K times per rollout (one predicted anchor per step), embarrassingly parallel.

## Data and I/O Requirements (WP3)

Datasets (project-wide list lives in WP1; WP3 consumes the converted robot shards):

- LIBERO (public, sim, embedded GT depth): 250 GB HDF5, 40 files, ~4,000 demos. Used for the
  feasibility runs (one-demo smoke file) and for post-training/eval. Ready.
- Open X-Embodiment subsets in LeRobot format (bridge, droid, fractal, taco_play, ...): GAM's
  pretraining mixture, ~784K trajectories with MimicGen and RoboCasa365. Sizes TODO from WP1.
- DROID 1.0.1, AgiBot World (TODO if in scope): sizes TODO.
- Depth pseudo-labels from the DA3 teacher (WP1.2 output), TODO TB.

Per-workload I/O profile, GAM training: input = LeRobot/HDF5 shards on `/iopsstor/scratch`
(read ~ samples/s x 8 anchors x 2 views x 256x256x3 uint8 + depth = TODO MB/s at 64 GPUs);
transient = none beyond logs; checkpoints = 14.2 GB per checkpoint (1.4B params + optimizer state,
as the released `pretrained-gam.pt`), every TODO steps, TODO kept on `/capstor/scratch`,
TODO on `/capstor/store`; retained = final checkpoints + eval outputs, TODO TB.

## Software infrastructure (WP3.1 workload)

Runtime: Container Engine EDF `env/gam.toml` = `nvcr.io#nvidia/pytorch:25.03-py3` (aarch64,
torch 2.7, Python 3.12) + venv `env/gam-venv` (build script `scripts/build_env.sh`, freeze
`env/gam-venv-freeze.txt`). Hooks: aws-ofi-nccl (cuda12), DCGM (for GSSR).

| Component (repo, version)                                                   | Purpose                                  | Ready?                         | Features / customizations                                                                     |
| --------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------- |
| GAM (`cvlab-kaist/Geometric-Action-Model` @ 58bb91e)                        | training + inference code                | yes                            | DDP via torchrun; DeepSpeed ZeRO-2 optional; one-line patch: `s/step` printed with 3 decimals |
| Depth-Anything-3 (`ByteDance-Seed/Depth-Anything-3` @ 2c21ea8)              | DA3-Giant backbone source                | yes                            | source on `PYTHONPATH`, no install; optional deps stubbed by GAM                              |
| Track4World DA3-Giant checkpoint (`SeonghuJeon/3da-libero-training-assets`) | backbone init, 5.5 GB                    | yes                            |                                                                                               |
| PyTorch 2.7 (NGC 25.03)                                                     | framework, flex_attention, torch.compile | yes                            | bf16 autocast, TF32 matmul                                                                    |
| NCCL (NGC) + aws-ofi-nccl hook                                              | multi-node collectives                   | yes                            | CSCS defaults (`FI_CXI_DISABLE_HOST_REGISTER`, `FI_MR_CACHE_MONITOR`)                         |
| transformers 5.5.4, timm 1.0.26, h5py 3.16, omegaconf 2.3                   | CLIP text encoder, data, config          | yes                            | HF offline, local CLIP folder                                                                 |
| deepspeed 0.18.8                                                            | ZeRO-2 alternative                       | installed, untested on aarch64 | TODO one run with `DS=1`                                                                      |
| GSSR (`eth-cscs/GPU-Saturation-Scorer` 2.0)                                 | GPU metrics + PDF                        | yes                            | built on login node, runs in container                                                        |
| WP1 data converters (LeRobot -> GAM pretraining loader)                     | production data                          | partial                        | GAM ships the OXE/MimicGen/RoboCasa loaders; consortium data TODO (WP1 timeline)              |

## Open decisions (need Philipp / Marc)

1. WP3.1 production config: views (2 or 3), history H and horizon, resolution (224 vs 252/448),
   and whether the WP2 checkpoint replaces Track4World DA3 as the init. Each changes tokens per
   sample and must be measured (4-GPU debug run, 15 min each).
2. Data target: which trajectory sets and how many windows per trajectory; number of full runs
   and seeds; ablation budget (template: avoid full-scale ablations).
3. WP3.2 K (candidate action chunks per sample) and the uncertainty head type; WP3.3 rollout
   count per model and evaluation matrix.
4. Whether 0.22 G is a target to fill or an output of this bottom-up budget.
