#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3 src/main.py \
  --reset_output_dir \
  --no_load_model \
  --train_set="exp6_v1" \
  --output_dir="outputs_exp6_v1" \
  --log_every=250 \
  --eval_every=5000 \
  "$@"

