#!/bin/bash
#SBATCH --job-name=gam-scale
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/veichta/rpr-wp3/runs/%x-%j.out
# One measurement job of GAM training. All model/data settings live in the CONFIG yaml; this script only
# sets the run length, the batch layout per node count, the launcher, and the profiling switches.
#   sbatch -N <nodes> [-t hh:mm:ss] --export=ALL,TAG=<tag>[,STEPS=60][,CONFIG=<yaml>][,GLOBAL=16][,MICRO=1]
#                     [,DS=1][,PROFILE=1][,EXTRA="--set k=v ..."] scripts/gam_scale.sbatch
#   GLOBAL = global batch held fixed across node counts (strong scaling): accum = GLOBAL / (MICRO * 4 * nodes).
#   DS=1   = DeepSpeed ZeRO-2 (JSON generated from MICRO/accum); default torchrun DDP.
#   PROFILE=1 = GAM phase timer every 10 steps + torch.profiler on micro-steps 30-35 (rank 0).
set -eo pipefail
source ${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}/scripts/env.sh
CONFIG=${CONFIG:-$R/configs/wp31_contract_518.yaml}
STEPS=${STEPS:-60}
TAG=${TAG:-scale}
MICRO=${MICRO:-1}
GLOBAL=${GLOBAL:-16}
NGPU=$((SLURM_NNODES * 4))
ACCUM=$((GLOBAL / (MICRO * NGPU)))
if [ "$ACCUM" -lt 1 ] || [ $((ACCUM * MICRO * NGPU)) -ne "$GLOBAL" ]; then
  echo "GLOBAL=$GLOBAL not divisible by MICRO*NGPU=$((MICRO * NGPU))"; exit 1
fi
OUT=$R/runs/${TAG}_${NGPU}gpu_${SLURM_JOB_ID}; mkdir -p $OUT
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 10000))
DSFLAG=""
if [ "${DS:-0}" = "1" ]; then
  # GAM hands its own AdamW to deepspeed.initialize; the JSON only carries batch layout + ZeRO stage.
  cat > $OUT/deepspeed.json <<EOF
{"train_micro_batch_size_per_gpu": $MICRO, "gradient_accumulation_steps": $ACCUM, "gradient_clipping": 1.0,
 "steps_per_print": 1000, "bf16": {"enabled": true}, "wall_clock_breakdown": false,
 "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "none"}, "contiguous_gradients": true,
                       "overlap_comm": true, "reduce_scatter": true, "reduce_bucket_size": 5e7, "allgather_bucket_size": 5e7}}
EOF
  DSFLAG="--deepspeed_config $OUT/deepspeed.json"
fi
if [ "${PROFILE:-0}" = "1" ]; then
  export DA3_STEP_PROFILE=1 DA3_STEP_PROFILE_EVERY=10 DA3_TORCH_PROFILE=30:35
fi
cp $CONFIG $OUT/config.yaml
echo "job $SLURM_JOB_ID nodes $SLURM_NNODES gpus $NGPU micro $MICRO accum $ACCUM global $GLOBAL steps $STEPS config $CONFIG ds ${DS:-0} profile ${PROFILE:-0} extra ${EXTRA:-}"
$SRUN bash -c "
  source $R/env/gam-venv/bin/activate &&
  $R/tools/gssr/gssr-record -o $OUT/gssr_report python -m torch.distributed.run \
    --nproc_per_node=4 --nnodes=\$SLURM_NNODES --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/train_robot.py --config $CONFIG $DSFLAG --results-dir $OUT/results \
    --set training.max_steps=$STEPS --set training.global_batch_size=$GLOBAL \
    --set training.micro_batch_size=$MICRO --set training.grad_accum_steps=$ACCUM ${EXTRA:-}
"

# GSSR PDFs for this run and any earlier run still lacking one (gssr-analyze needs the container venv).
# Plain srun here: $SRUN ends in numactl, which would swallow the --ntasks/--nodes flags.
srun -A a144 -ul --ntasks=1 --nodes=1 --environment=$R/env/gam.toml --container-workdir=$R/code/gam bash $R/scripts/gssr_pdfs.sh
