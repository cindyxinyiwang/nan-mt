#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3 src/main.py \
  --clean_mem_every=5 \
  --output_dir="outputs_exp4_v9" \
  --log_every=100 \
  --eval_every=2000 \
  --reset_output_dir \
  --no_load_model \
  --data_path="data/bpe_32k_shared/en-de/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="dev2010.en" \
  --target_valid="dev2010.de" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --share_emb_and_softmax \
  --cuda \
  --batch_size=128 \
  --batcher="sent" \
  --loss_norm="sent" \
  --n_train_sents=250000 \
  --max_len=350 \
  --d_word_vec=288 \
  --d_model=288 \
  --d_inner=507 \
  --n_layers=5 \
  --d_k=64 \
  --d_v=64 \
  --n_heads=3 \
  --n_train_steps=200000 \
  --n_warm_ups=2000 \
  --dropout=0.18 \
  --lr_adam=0.0003 \
  --lr_sgd=0.25 \
  --l2_reg=1e-6 \
  --optim="adam" \
  --optim_switch=100000 \
  --lr_dec=1.1 \
  --init_range=0.04 \
  --grad_bound=5.0 \
  --init_type="uniform" \
  "$@"

