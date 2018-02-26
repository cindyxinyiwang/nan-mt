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

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp2_v2" \
#   --output_dir="outputs_exp2_v2" \
#   --log_every=250 \
#   "$@"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp3_v1" \
#   --output_dir="outputs_exp3_v1" \
#   --log_every=250 \
#   "$@"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp3_v2" \
#   --output_dir="outputs_exp3_v2" \
#   --log_every=250 \
#   "$@"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp4_v1" \
#   --output_dir="outputs_exp4_v1" \
#   --log_every=50 \
#   "$@"

# python3 src/main.py \
#   --reset_output_dir \
#   --no_load_model \
#   --train_set="exp5_v1" \
#   --output_dir="outputs_exp5_v1" \
#   --log_every=100 \
#   "$@"

python3 src/main.py \
  --reset_output_dir \
  --no_load_model \
  --train_set="exp5_v2" \
  --output_dir="outputs_exp5_v2" \
  --log_every=100 \
  "$@"

