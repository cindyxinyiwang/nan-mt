#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"

python src/main.py \
  --output_dir="outputs" \
  --log_every=5 \
  "$@"

