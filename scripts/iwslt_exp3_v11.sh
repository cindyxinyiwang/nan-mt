#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3 src/main.py \
  --clean_mem_every=5 \
  --output_dir="outputs_exp3_v11" \
  --log_every=100 \
  --eval_every=2000 \
  --no_reset_output_dir \
  --load_model \
  --data_path="data/bpe_24k_shared/en-de/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="dev2010.en" \
  --target_valid="dev2010.de" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --batch_size=48 \
  --n_train_sents=250000 \
  --max_len=300 \
  --d_word_vec=384 \
  --d_model=384 \
  --d_inner=512 \
  --n_layers=5 \
  --d_k=64 \
  --d_v=64 \
  --n_heads=6 \
  --n_train_steps=150000 \
  --n_warm_ups=4000 \
  --dropout=0.2 \
  --share_emb_and_softmax \
  --cuda \
  --lr=0.001 \
  --lr_dec=1.15 \
  --init_range=0.04 \
  --grad_bound=5.0 \
  "$@"

