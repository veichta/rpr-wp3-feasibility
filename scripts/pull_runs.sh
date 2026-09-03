#!/bin/bash
# Pull job outputs (slurm .out, GSSR PDFs + raw reports, training logs) from Clariden to the local mirror.
# Run from the local mirror folder: bash scripts/pull_runs.sh
set -e
R=${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}
rsync -a --timeout=120 --include='*/' --include='*.out' --include='*.pdf' --include='*.log' --include='*.json' --include='*.csv' --include='*.txt' --exclude='*' clariden:$R/runs/ runs/
find runs -type d -empty -delete 2>/dev/null; ls -R runs | head -40
