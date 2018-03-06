#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

# This experiment is to show that the model can overfit DEV data.
# BLEU: 94.26

python3.6 src/main.py \
  --output_dir="outputs_exp1_v1" \
  --log_every=100 \
  --eval_every=1000 \
  --reset_output_dir \
  --no_load_model \
  --data_path="data/bpe_28k_shared/en-de/" \
  --source_train="dev2010.en" \
  --target_train="dev2010.de" \
  --source_valid="dev2010.en" \
  --target_valid="dev2010.de" \
  --source_vocab="shared_28000.vocab" \
  --target_vocab="shared_28000.vocab" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --batch_size=2 \
  --n_train_sents=25000 \
  --max_len=1500 \
  --d_word_vec=128 \
  --d_model=128 \
  --d_inner=192 \
  --n_layers=5 \
  --d_k=32 \
  --d_v=32 \
  --n_heads=4 \
  --n_train_steps=5000 \
  --n_warm_ups=750 \
  --dropout=0.0 \
  "$@"

