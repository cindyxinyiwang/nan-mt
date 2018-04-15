#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

# just modify the path to the model

python3.6 src/translate.py \
  --model_dir="outputs_exp10_v3" \
  --data_path="data/clean_piece_37k_shared/en-de/" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="tst2013.en" \
  --target_test="tst2013.de" \
  --merge_bpe \
  --batch_size=32 \
  --beam_size=4 \
  --max_len=600 \
  --n_train_sents=10000 \
  --cuda \
  --no_raml_source \
  --out_file="trans" \
  --n_corrupts=5 \
  "$@"

