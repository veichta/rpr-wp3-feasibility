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

| #GPUs | nodes | micro x accum | s/step | samples/s | efficiency | SLURM job | GSSR PDF |
| ----- | ----- | ------------- | ------ | --------- | ---------- | --------- | -------- |
| 4 | 1 | 3 x 4 | 3.333 (std 0.287, 1000 steps) | 14.40 | 100 % | 3277872 | runs/long_4gpu_3277872/gssr_3277872.pdf: GPU util 66 %, FP 0.17, 373 W |
| 8 | 2 | 3 x 2 | 1.657 (std 0.195, 1200 steps) | 28.97 | 101 % | 3277945 | runs/long_8gpu_3277945/gssr_3277945.pdf |
| 12 | 3 | 2 x 2 | 1.282 (std 0.116, 1071 steps; one 121 s loader stall at an epoch boundary excluded) | 37.4 | 87 % | 3278593 | runs/long_12gpu_3278593/gssr_3278593.pdf |
| 16    | 4     | 3 x 1         | TODO   | TODO      | TODO       | TODO      | TODO     |

Beyond 16 GPUs (debug partition ceiling) requires the `normal` partition: TODO 32/64 GPU points
if the queue allows before submission; otherwise state the 16-GPU measurement and the DDP
communication argument (1B trainable params, 4 GB bf16 gradient all-reduce per step, overlapped
with backward).

GSSR (all runs, training phase): ~70 % GPU utilization, ~55 % SM active, ~20 % SM occupancy,
~15-18 % tensor-pipe activity, ~370 W per GPU. Reports: `runs/<run>/gssr_<jobid>.pdf`, raw
DCGM CSVs in `runs/<run>/gssr_report/`. Logs and job scripts: TODO copy to
`/capstor/store/cscs/swissai/a144/Logs/feasibility_review_rpr/WP3/` (csstaff-readable).

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
