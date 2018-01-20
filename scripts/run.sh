#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/main.py \
  --output_dir="outputs" \
  --data_dir="data" \
  "$@"

