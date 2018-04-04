#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3.6 src/main.py \
  --clean_mem_every=5 \
  --output_dir="outputs_exp10_v3" \
  --log_every=100 \
  --eval_every=2000 \
  --reset_output_dir \
  --no_load_model \
  --data_path="data/clean_piece_37k_shared/en-de/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="tst2013.en" \
  --target_valid="tst2013.de" \
  --target_valid_ref="data/preproc/en-de/tst2013.truecase.de" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --share_emb_and_softmax \
  --cuda \
  --batch_size=3000 \
  --batcher="word" \
  --loss_norm="sent" \
  --n_train_sents=250000 \
  --max_len=1000 \
  --d_word_vec=512 \
  --d_model=512 \
  --d_inner=768 \
  --n_layers=6 \
  --d_k=64 \
  --d_v=64 \
  --n_heads=8 \
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
  --raml_source \
  --raml_target \
  --raml_tau=0.75 \
  "$@"

