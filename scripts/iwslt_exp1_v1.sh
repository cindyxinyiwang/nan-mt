#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3 src/main.py \
  --output_dir="outputs_exp1_v1" \
  --log_every=100 \
  --eval_every=1000 \
  --reset_output_dir \
  --no_load_model \
  --data_path="data/bpe_28k_shared/en-de/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="dev2010.en" \
  --target_valid="dev2010.de" \
  --source_vocab="shared_28000.vocab" \
  --target_vocab="shared_28000.vocab" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --batch_size=32 \
  --n_train_sents=250000 \
  --max_len=600 \
  --d_word_vec=256 \
  --d_model=256 \
  --d_inner=384 \
  --n_layers=5 \
  --d_k=64 \
  --d_v=64 \
  --n_heads=4 \
  --n_train_steps=100000 \
  --n_warm_ups=750 \
  --dropout=0.1 \
  --cuda \
  "$@"

