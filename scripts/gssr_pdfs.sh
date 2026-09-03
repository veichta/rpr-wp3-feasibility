#!/bin/bash
# Make a GSSR PDF for every runs/*/gssr_report that lacks one. Run INSIDE the container (pandas/matplotlib).
R=${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}
source $R/env/gam-venv/bin/activate
for d in $R/runs/*/gssr_report; do
  run=$(dirname $d); jid=${run##*_}
  [ -f $run/gssr_$jid.pdf ] && continue
  GSSR_UV_ACTIVE=1 python $R/tools/gssr/gssr-analyze $d -o $run/gssr_$jid.pdf && echo "pdf $run/gssr_$jid.pdf"
done
