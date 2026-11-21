#!/usr/bin/env bash
source /home/dcs/anaconda3/etc/profile.d/conda.sh
conda activate embodiedgen
export CUDA_VISIBLE_DEVICES=4 GPT_PROVIDER=codex MODEL_NAME=gpt-6-astra GPT_TIMEOUT=180
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
