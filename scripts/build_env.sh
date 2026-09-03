#!/bin/bash
# Build the GAM training venv on top of the NGC PyTorch 25.03 container. Run INSIDE the container.
# = GAM requirements.txt minus the MuJoCo/robosuite/LIBERO closed-loop eval stack (training only).
set -euo pipefail
R=${RPR_WP3_ROOT:-/iopsstor/scratch/cscs/veichta/rpr-wp3}
export PIP_CONSTRAINT=""   # NGC pins (regex etc.) via PIP_CONSTRAINT; the venv needs newer
rm -rf $R/env/gam-venv
python -m venv --system-site-packages $R/env/gam-venv
source $R/env/gam-venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir \
  transformers==5.5.4 tokenizers==0.22.2 huggingface_hub==1.10.1 safetensors==0.5.3 accelerate==1.13.0 \
  datasets==4.8.4 timm==1.0.26 einops==0.8.1 deepspeed==0.18.8 h5py==3.16.0 omegaconf==2.3.0 \
  antlr4-python3-runtime==4.9.3 opencv-python-headless==4.11.0.86 imageio==2.37.3 imageio-ffmpeg==0.6.0 \
  av==15.1.0 plotly==5.24.1 wandb==0.24.2 protobuf==4.24.4 jsonlines
python $R/scripts/check_env.py
pip freeze > $R/env/gam-venv-freeze.txt
