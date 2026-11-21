#!/usr/bin/env bash
# Source from the repository root. No keys or credential-file access.
source /home/dcs/anaconda3/etc/profile.d/conda.sh
conda activate embodiedgen
export CUDA_VISIBLE_DEVICES=7
export GPT_PROVIDER=codex
export MODEL_NAME=gpt-6-astra
export GPT_TIMEOUT=180
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
