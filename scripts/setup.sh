#!/bin/bash
# One-shot setup on a Clariden LOGIN node (needs internet): code at pinned commits, symlinks, assets, GSSR.
# Idempotent: re-running skips what already exists. Root = $RPR_WP3_ROOT (default below).
set -euo pipefail
R=${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}
GAM_COMMIT=58bb91e                                     # cvlab-kaist/Geometric-Action-Model, 2026-08-23
DA3_COMMIT=2c21ea849ceec7b469a3e62ea0c0e270afc3281a   # ByteDance-Seed/Depth-Anything-3, pinned by GAM Dockerfile
mkdir -p $R/code $R/env $R/assets $R/tools $R/runs

# --- code -------------------------------------------------------------------
if [ ! -d $R/code/gam ]; then
  git clone -q https://github.com/cvlab-kaist/Geometric-Action-Model $R/code/gam
  git -C $R/code/gam checkout -q $GAM_COMMIT
fi
if [ ! -d $R/code/Depth-Anything-3 ]; then
  git init -q $R/code/Depth-Anything-3
  git -C $R/code/Depth-Anything-3 remote add origin https://github.com/ByteDance-Seed/Depth-Anything-3.git
  git -C $R/code/Depth-Anything-3 fetch -q --depth 1 origin $DA3_COMMIT
  git -C $R/code/Depth-Anything-3 checkout -q FETCH_HEAD
fi
# GAM resolves DA3, data and checkpoints relative to its own root (DA3_ROOT = code/gam).
ln -sfn ../Depth-Anything-3 $R/code/gam/Depth-Anything-3
ln -sfn $R/assets/checkpoints $R/code/gam/checkpoints
ln -sfn $R/assets/data $R/code/gam/data
# Local patch: GAM prints s/step with one decimal, too coarse at 16+ GPUs (0.7 s/step). Print three.
sed -i "s#s/step=%.1f#s/step=%.3f#" $R/code/gam/src/train_robot.py
echo "code: gam $(git -C $R/code/gam rev-parse --short HEAD) (+s/step %.3f patch), da3 $(git -C $R/code/Depth-Anything-3 rev-parse --short HEAD)"

# --- assets (Hugging Face via plain curl; skips complete files, resumes partial ones) ---
hf_get () {  # url -> local path
  mkdir -p "$(dirname "$2")"
  curl -sL --retry 5 -C - -o "$2" "$1"
  echo "asset $2 $(stat -c %s "$2") bytes"
}
REPO=https://huggingface.co/datasets/SeonghuJeon/3da-libero-training-assets/resolve/main
API=https://huggingface.co/api/datasets/SeonghuJeon/3da-libero-training-assets/tree/main
for prefix in data/libero_noop/_stats data/libero_noop/_smoke_libero_spatial; do
  for p in $(curl -s "$API/$prefix" | python3 -c "import sys,json; [print(x[\"path\"]) for x in json.load(sys.stdin) if x[\"type\"]==\"file\"]"); do
    hf_get $REPO/$p $R/assets/$p
  done
done
hf_get $REPO/checkpoints/track4world_da3.pth $R/assets/checkpoints/track4world_da3.pth
for f in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors; do
  hf_get https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/$f $R/assets/clip-vit-large-patch14/$f
done

# --- GSSR (GPU Saturation Scorer, C + DCGM; built on the login node, runs inside the container) ---
if [ ! -x $R/tools/gssr/gssr-record ]; then
  git clone -q --depth 1 https://github.com/eth-cscs/GPU-Saturation-Scorer $R/tools/gssr
  make -C $R/tools/gssr
fi
echo "setup done: $R"
