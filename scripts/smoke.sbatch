#!/bin/bash
#SBATCH --job-name=gam-smoke
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/veichta/rpr-wp3/runs/%x-%j.out
# Smoke: 60 s GPU burn under gssr-record + gssr-analyze PDF, then a 3-step single-GPU GAM training step.
set -eo pipefail
source ${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}/scripts/env.sh
OUT=$R/runs/smoke_$SLURM_JOB_ID; mkdir -p $OUT
$SRUN bash -c "
  set -e
  source $R/env/gam-venv/bin/activate
  echo == gssr burn; $R/tools/gssr/gssr-record -o $OUT/gssr_burn python $R/scripts/gpu_burn.py
  echo == gssr analyze; GSSR_UV_ACTIVE=1 python $R/tools/gssr/gssr-analyze $OUT/gssr_burn -o $OUT/gssr_burn.pdf && ls -la $OUT/gssr_burn.pdf
  echo == gam smoke 1 gpu; python src/train_robot.py --config configs/training/libero_unified/smoke/gam_chunk2.yaml --single-gpu --results-dir $OUT/results --set training.max_steps=3 --set predictor.clip_model=$CLIP
"
