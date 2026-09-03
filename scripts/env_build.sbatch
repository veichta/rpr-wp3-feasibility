#!/bin/bash
#SBATCH --job-name=gam-envbuild
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/veichta/rpr-wp3/runs/%x-%j.out
# Build env/gam-venv inside the container and verify the imports. ~10 min.
set -eo pipefail
source ${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}/scripts/env.sh
$SRUN bash -c "bash $R/scripts/build_env.sh"
