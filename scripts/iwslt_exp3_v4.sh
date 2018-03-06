#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3 src/main.py \
  --clean_mem_every=5 \
  --output_dir="outputs_exp3_v4" \
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
  --n_train_sents=250000 \
  --max_len=250 \
  --d_word_vec=512 \
  --d_model=512 \
  --d_inner=1024 \
  --n_layers=6 \
  --d_k=64 \
  --d_v=64 \
  --n_heads=8 \
  --n_train_steps=80000 \
  --n_warm_ups=2000 \
  --dropout=0.1 \
  --share_emb_and_softmax \
  --cuda \
  --lr=20.0 \
  --lr_dec=1.5 \
  --init_range=0.08 \
  --grad_bound=0.10 \
  "$@"

