#!/bin/bash

#This script runs iwlst experiment with Jiatao Gu's hparam

python src/main.py \
  --clean_mem_every=5 \
  --output_dir="outputs_gu" \
  --log_every=100 \
  --eval_every=1000 \
  --no_load_model \
  --reset_output_dir \
  --data_path="data/bpe_24k_shared/en-de/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="dev2010.en" \
  --target_valid="dev2010.de" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --batcher="word" \
  --loss_norm="word" \
  --batch_size=2048 \
  --n_train_sents=250000 \
  --max_len=300 \
  --d_word_vec=288 \
  --d_model=288 \
  --d_inner=507 \
  --n_layers=5 \
  --d_k=64 \
  --d_v=64 \
  --n_heads=2 \
  --n_train_steps=1000000 \
  --n_warm_ups=746 \
  --dropout=0.079 \
  --share_emb_and_softmax \
  --lr_dec=1 \
  --lr_schedule \
  --optim="adam" \
  --init_type="xavier_normal" \
  --seed 19920206 \
  "$@"

