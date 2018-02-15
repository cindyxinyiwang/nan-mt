#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3 src/main.py \
  --output_dir="outputs" \
  --log_every=250 \
  "$@"

