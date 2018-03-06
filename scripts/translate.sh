#!/bin/bash

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

# just modify the path to the model

python3.6 src/translate.py \
  --model_dir="outputs_exp3_v2" \
  --data_path="data/bpe_24k_shared/en-de/" \
  --source_vocab="vocab.en" \
  --target_vocab="vocab.de" \
  --source_test="dev2010.en" \
  --target_test="dev2010.de" \
  --merge_bpe \
  --batch_size=32 \
  --beam_size=4 \
  --max_len=750 \
  --n_train_sents=10000 \
  --cuda \
  "$@"

