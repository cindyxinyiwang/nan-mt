#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"

python3.6 src/main.py \
  --clean_mem_every=5 \
  --output_dir="outputs_exp11_v18" \
  --log_every=100 \
  --eval_every=2000 \
  --no_load_model \
  --reset_output_dir \
  --data_path="data/raw/de-en/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="valid.en" \
  --target_valid="valid.de" \
  --target_valid_ref="data/raw/de-en/valid.de" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="test.en" \
  --target_test="test.de" \
  --share_emb_and_softmax \
  --batch_size=4500 \
  --batcher="word" \
  --loss_norm="sent" \
  --n_train_sents=250000 \
  --max_len=1000 \
  --d_word_vec=288 \
  --d_model=288 \
  --d_inner=507 \
  --n_layers=5 \
  --d_k=64 \
  --d_v=64 \
  --n_heads=5 \
  --n_train_steps=350000 \
  --n_warm_ups=2000 \
  --dropout=0.25 \
  --lr_adam=0.001 \
  --lr_sgd=0.05 \
  --optim_switch=250000 \
  --optim="adam" \
  --lr_dec=1.05 \
  --init_range=0.04 \
  --grad_bound=5.0 \
  --raml_target \
  --raml_tau=0.75 \
  --cuda \
  "$@"

