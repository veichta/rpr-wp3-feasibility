#!/bin/bash
#SBATCH --job-name=gam-scale
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/veichta/rpr-wp3/runs/%x-%j.out
# One strong-scaling point of GAM training (paper config, LIBERO smoke data, global batch fixed at 48).
# The smoke file holds one demo: min_train_batches_per_epoch virtualizes the epoch (random window start per sample).
#   sbatch -N <nodes> [-t hh:mm:ss] --export=ALL,TAG=<tag>[,STEPS=300][,CONFIG=<yaml>][,DS=1][,EXTRA="--set k=v ..."] scripts/gam_scale.sbatch
set -eo pipefail
source ${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}/scripts/env.sh
CONFIG=${CONFIG:-configs/training/libero_unified/gam/chunk8_150k_2node.yaml}
STEPS=${STEPS:-300}
TAG=${TAG:-scale}
NGPU=$((SLURM_NNODES * 4))
case $NGPU in       # micro x accum x gpus = 48
  4)  MICRO=3; ACCUM=4 ;;
  8)  MICRO=3; ACCUM=2 ;;
  12) MICRO=2; ACCUM=2 ;;
  16) MICRO=3; ACCUM=1 ;;
  *)  echo "unsupported gpu count $NGPU"; exit 1 ;;
esac
MICRO=${MICRO_OVERRIDE:-$MICRO}; ACCUM=${ACCUM_OVERRIDE:-$ACCUM}; GLOBAL=${GLOBAL_OVERRIDE:-48}   # e.g. MICRO_OVERRIDE=1 ACCUM_OVERRIDE=12 keeps global 48 on 4 GPUs
OUT=$R/runs/${TAG}_${NGPU}gpu_${SLURM_JOB_ID}; mkdir -p $OUT
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 10000))
# DS=1: DeepSpeed ZeRO-2 instead of DDP. GAM hands its own AdamW to deepspeed.initialize, so the JSON only sets
# batch/accumulation (must match micro x accum above for the 4-GPU point) and the ZeRO stage.
DSFLAG=""; [ "${DS:-0}" = "1" ] && DSFLAG="--deepspeed_config ${DS_CONFIG:-$R/scripts/ds_micro3_accum4.json}"
echo "job $SLURM_JOB_ID nodes $SLURM_NNODES gpus $NGPU micro $MICRO accum $ACCUM steps $STEPS config $CONFIG ds ${DS:-0} extra ${EXTRA:-}"
$SRUN bash -c "
  source $R/env/gam-venv/bin/activate &&
  $R/tools/gssr/gssr-record -o $OUT/gssr_report python -m torch.distributed.run \
    --nproc_per_node=4 --nnodes=\$SLURM_NNODES --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/train_robot.py --config $CONFIG $DSFLAG --results-dir $OUT/results \
    --set training.max_steps=$STEPS --set training.log_every=10 --set training.ckpt_every=999999 --set training.vis_every=999999 \
    --set training.global_batch_size=$GLOBAL --set training.micro_batch_size=$MICRO --set training.grad_accum_steps=$ACCUM \
    --set dataset.eval_ratio=0 --set training.min_train_batches_per_epoch=200 --set predictor.clip_model=$CLIP ${EXTRA:-}
"

# GSSR PDFs for this run and any earlier run still lacking one (gssr-analyze needs the container venv).
# Plain srun here: $SRUN ends in numactl, which would swallow the --ntasks/--nodes flags.
srun -A a144 -ul --ntasks=1 --nodes=1 --environment=$R/env/gam.toml --container-workdir=$R/code/gam bash $R/scripts/gssr_pdfs.sh
