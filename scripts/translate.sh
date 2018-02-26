#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3 src/translate.py \
  --model_dir="outputs_exp5_v2" \
  "$@"

