# WP3 feasibility review: what goes into the report

Working notes for writing the WP3 part of the Swiss AI call-4 Feasibility Review (template in
`../references/feasibility_review_template_call4.txt`; accepted examples: Alahi RGWM
`../references/RealityGrounded_World_Model_Feasibility_review.docx`, Zamir/Tang/Fleuret
`../references/SwissAI_March2026_feasibility_report_final_gssr_rasterized_compressed-1.pdf`).
Measured numbers live in `README.md` (job log) and `WP3_feasibility.md` (draft tables).

## 0. Framing sentences to reuse

- WP3 = action-conditioned geometric world modeling on the GFM latent space. Production stack: GAM
  (Han et al. 2026, consortium paper: Hong, Pollefeys, Hutter) = DA3-Giant split at block 13, causal
  future predictor, geometry heads. 1.4B params, all trainable.
- Contract (WP3.1 model shape): 448 x 448 inputs (pi0.7 operating point; pi0.7 resizes every source to
  448 and masks missing cameras), up to 3 views, 16 anchors at 8-frame spacing (6.4 s window),
  full fine-tune, depth + ray + point-map losses, bf16, ZeRO-2, micro-batch 1.
- Differentiator vs pi0.7: their "world model" generates pixel subgoals from text (BAGEL 14B), not
  action-conditioned, not geometric; WP3 predicts consequences of candidate actions in geometry.
- Honesty statements to include verbatim somewhere:
  1. Input for all measurements was a cached single LIBERO demo (compute + communication profile
     only); data-loader wait logged 0.00 s; I/O requirements derived analytically (Sec. 4).
  2. Probe stand-ins, compute-neutral: third view = repeated external camera; anchor spacing 4
     frames instead of 8 (98-frame demo). Production spacing changes frame indices, not tokens.
  3. The K-step rollout training loop (multi-horizon supervision) is in development (software table:
     "No, ETA M1"); measured pass = GAM's next-anchor pass with the identical model and data shape;
     rollout cost lies within 0.7-1.5x of it (predictor re-runs vs smaller deep pass).
  4. Per-rank history-length sampling in the public GAM code causes load imbalance (49% of GPU time
     in NCCL waits); fixed by a rank-synchronized draw (`patches/gam_sync_H.py`). Report the fixed
     version; mention the finding as resource-efficiency work.

## 1. Resource Justification and Timeline (WP3 rows)

Template table columns: WP [deps] | target #GPUs/job | #runs, #concurrent | data target | benchmarked?
diff-to-prod? data ready? | avg throughput [strong-scaling eff %] | run duration, WP duration (buffer) | GPUh.

- WP3.1 training. Throughput: 0.140 samples/s/GPU = 6.9k visual tokens/s/GPU (job 3282051, 4 GPU);
  ladder numbers TODO (jobs `ladder`). Target scale: 64 GPUs (16 nodes) TODO decide; ideal accum =
  1 at target. Data target = windows x epochs: TODO from WP1 (DROID 76K traj, AgiBot ~1M, RH20T
  110K, BridgeV2/RT-1 raw ~130K, OXE remainder + sim ~500K, consortium TBD) -> placeholder 100M
  windows x 2-3 epochs. GPU-h = windows x epochs / (0.140 x 3600) / eff + 15% overhead.
  Experiment table (Zamir style): rows = design sweeps at 4-16 GPU (short), validation runs, final
  run(s) at target scale, ablations bounded (<= 25% of the final run each). Wall-clock check: the
  final run at 64 GPUs must fit inside the WP window with pro-rata monthly usage.
- WP3.2 counterfactuals + uncertainty (post-training on a subset). Cost = N candidates x WP3.1
  per-sample cost (a candidate chunk is an action input: same pass) + ensemble M x predictor share
  (predictor = ~21% of forward, phase timer job 3281372). Placeholder N=4, M=3, 25% of windows.
- WP3.3 rollout inference for planning / imagined experience. Cost = rollouts x K steps x forward
  cost per predicted anchor; measured forward-only throughput TODO (job `infer`, `--eval-only`),
  target-scale GSSR at 16 GPUs (embarrassingly parallel). Placeholder 20M rollouts x 8 steps.
- Split placeholder 55 / 25 / 20 of 0.22 G = 1.01M GPU-h; decide with Philipp/Marc whether 0.22 G
  is a target or an output of the sum.
- Gantt: M1-M2 rollout loop + data converters (dep. WP1), M3-M8 WP3.1 (dep. WP2 checkpoint at M3,
  else Track4World DA3 init), M6-M10 WP3.2, M8-M12 WP3.3 feeding WP5. Add 5% restart margin
  (template) or 15% (Zamir) -- pick one and say which.

## 2. Application Efficiency on Alps

- Model/config paragraph: copy from `configs/wp31_contract_448.yaml` (tokens per sample formula:
  16 anchors x 3 views x (448/14)^2 = 49,152 visual tokens + 77 text + action/proprio tokens).
- Parallel setup: DP only, torchrun + DeepSpeed ZeRO-2, micro-batch 1 (memory ceiling 70 GB at
  448 with deep checkpointing), accumulation to hold the global batch fixed across the ladder;
  minimal setup = 1 GPU; baseline = 1 node.
- Ladder table with job IDs + log paths (`/capstor/store/cscs/swissai/a144/...` copy TODO):
  4/8/12/16 GPUs at global batch 48 (TODO numbers). Beyond 16 GPUs: `normal` partition TODO
  (32/64) or justify by the DDP argument (1.4B trainable, 2.8 GB bf16 gradients per step,
  reduce-scatter overlapped with backward; 98% compute occupancy measured).
- GSSR: contract 448 = GPU util 85-88% good, FP util 0.37-0.39 good, 441-452 W acceptable,
  GPU start 87% good (job 3282051). Add MFU: FLOPs/sample (analytic) / (7.13 s x 989 TFLOP/s) TODO.
- Profiling paragraph (short): torch.profiler + phase timer, where the time goes (deep global
  attention over 45 frames ~40%, backward 3x forward with recompute), the H-imbalance finding.
- Launcher check: ZeRO-2 vs DDP (84% at the paper config), why ZeRO-2 for the contract (memory).
- First-series numbers (paper config, 4-16 GPU, 100/101/87/96%) can go in an appendix as the
  small-scale sweep configuration.

## 3. Data and I/O Requirements

- Dataset table (availability on Alps, quality, TB, format, #files, sensitivity) for: DROID,
  AgiBot World, RH20T, BridgeV2/RT-1 raw, OXE LeRobot subsets, LIBERO, MimicGen, RoboCasa365,
  consortium captures, DA3 teacher depth pseudo-labels (WP1.2 output). TODO sizes from WP1.
- Per-workload I/O: bytes per sample at 448 (16 anchors x 3 views x 448x448x3 uint8 = 29 MB raw,
  ~2-3 MB as JPEG/WebP) x samples/s at target scale -> MB/s; video-decode CPU cost for the
  LeRobot mp4 sources (state as the real limiter, not bandwidth); shards on /iopsstor/scratch.
- Checkpoints: 1.4B params + optimizer state = 12-17 GB per checkpoint (measured dumps), every
  N steps, K kept on /capstor/scratch, finals on /capstor/store.
- Retained outputs: final checkpoints, eval outputs, WP3.3 rollout datasets for WP5 (size TODO).

## 4. Software Infrastructure

Table already in `WP3_feasibility.md`: NGC PyTorch 25.03 container + venv (freeze file), GAM
@58bb91e + 3 local patches, DA3 @2c21ea8, transformers 5.5.4, deepspeed 0.18.8, NCCL via
aws-ofi hook, GSSR 2.0, WP1 converters (partial), rollout loop (No, ETA M1), N-candidate batching
for WP3.2 (No, ETA M6).

## 5. Reproducibility package for csstaff

- Copy per run: slurm .out, config.yaml, gssr_<jid>.pdf, gssr_report/ CSVs, log.txt to
  `/capstor/store/cscs/swissai/a144/Logs/feasibility_review_rpr/WP3/<row>/` (readable by csstaff);
  list job IDs in the tables. Repo: github.com/veichta/rpr-wp3-feasibility (setup.sh reproduces
  code, assets, env, patches).

## 6. Open decisions (need Philipp / Marc)

1. Data target: which sources are staged, windows per episode, epochs.
2. Target scale (64 vs 128 GPUs) and whether to run the `normal`-partition points.
3. WP3.1 / 3.2 / 3.3 split and whether 0.22 G is the target.
4. Overhead convention (5% template vs 15% Zamir).
5. Whether the encoder init is the WP2 checkpoint or Track4World DA3 (affects the Gantt only).
