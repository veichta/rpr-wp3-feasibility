# Runtime environment for the GAM WP3 feasibility runs. Sourced by every sbatch script; exports propagate into srun/container.
export R=${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}
export DA3_ROOT=$R/code/gam
export DA3_BASE_CKPT=$DA3_ROOT/checkpoints/track4world_da3.pth
export CLIP=$R/assets/clip-vit-large-patch14
export PYTHONPATH=$DA3_ROOT/src:$R/code/Depth-Anything-3/src   # GAM only adds src/Depth-Anything-3/src itself
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TORCH_EXTENSIONS_DIR=$R/env/torch_extensions
export TRITON_CACHE_DIR=$R/env/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
# srun prefix shared by all jobs: container + DCGM hook, GAM as working dir, NUMA-local host memory.
export SRUN="srun -A a144 -ul --environment=$R/env/gam.toml --container-workdir=$R/code/gam numactl --membind=0-3"
