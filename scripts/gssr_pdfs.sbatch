#!/bin/bash
#SBATCH --job-name=gam-gssr-pdfs
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:08:00
#SBATCH --output=/iopsstor/scratch/cscs/veichta/rpr-wp3/runs/%x-%j.out
# Standalone PDF generation for runs whose own PDF stage did not run.
set -eo pipefail
source ${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}/scripts/env.sh
srun -A a144 -ul --ntasks=1 --nodes=1 --environment=$R/env/gam.toml --container-workdir=$R/code/gam bash $R/scripts/gssr_pdfs.sh
