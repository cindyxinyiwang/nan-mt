#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3 src/main.py \
  --output_dir="outputs_exp3_v1" \
  --log_every=100 \
  --eval_every=2000 \
  --reset_output_dir \
  --no_load_model \
  --data_path="data/bpe_24k_shared/en-de/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="dev2010.en" \
  --target_valid="dev2010.de" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --batch_size=32 \
  --n_train_sents=25000 \
  --max_len=750 \
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
  --share_emb_and_softmax \
  --cuda \
  --lr=20.0 \
  --init_range=0.08 \
  --grad_bound=0.05 \
  "$@"

