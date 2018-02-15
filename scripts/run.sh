#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3 src/main.py \
  --reset_output_dir \
  --no_load_model \
  --train_set="bpe32" \
  --output_dir="outputs" \
  --log_every=50 \
  "$@"

