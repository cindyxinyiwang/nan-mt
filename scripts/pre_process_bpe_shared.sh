#!/bin/bash

vocab_size=32000

languages=("en" "de")
file_names=(
  "train"
  "dev2010"
  "tst2010"
  "tst2011"
  "tst2012"
  "tst2013"
  "tst2014"
)

echo "Training BPE on train.{en,de}"
spm_train \
  --input="data/raw/en-de/train.en,data/raw/en-de/train.de" \
  --model_prefix="data/bpe_32k_shared_vocab/en-de/shared_${vocab_size}" \
  --vocab_size=${vocab_size} \
  --model_type="bpe"

for language in ${languages[@]}
do
  for file_name in ${file_names[@]}
  do
    echo "Generating BPE for ${file_name}.$language"
    spm_encode \
      --extra_options="eos" \
      --model="data/bpe_32k_shared_vocab/en-de/shared_${vocab_size}.model" \
      --output_format="piece" \
      < "data/raw/en-de/${file_name}.$language" \
      > "data/bpe_32k_shared_vocab/en-de/${file_name}.$language"
  done
done

