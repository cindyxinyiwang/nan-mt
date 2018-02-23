#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp1_v1" \
#   --output_dir="outputs_exp1_v1" \
#   --log_every=250 \
#   "$@"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp1_v2" \
#   --output_dir="outputs_exp1_v2" \
#   --log_every=250 \
#   "$@"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp1_v3" \
#   --output_dir="outputs_exp1_v3" \
#   --log_every=250 \
#   "$@"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp2_v1" \
#   --output_dir="outputs_exp2_v1" \
#   --log_every=250 \
#   "$@"

python3 src/main.py \
  --reset_output_dir \
  --no_load_model \
  --train_set="exp2_v2" \
  --output_dir="outputs_exp2_v2" \
  --log_every=250 \
  "$@"

