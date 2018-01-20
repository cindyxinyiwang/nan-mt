#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/pre_process.py \
  "$@"

