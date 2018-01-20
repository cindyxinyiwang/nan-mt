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
for language in ${languages[@]}
do
  echo "Training BPE on train.$language"
  spm_train \
    --input="data/en-de/train.$language" \
    --model_prefix="data/en-de/$language.bpe.${vocab_size}" \
    --vocab_size=${vocab_size} \
    --model_type="bpe"

  for file_name in ${file_names[@]}
  do
    echo "Generating BPE for ${file_name}.$language"
    spm_encode \
      --extra_options="eos" \
      --model="data/en-de/$language.bpe.${vocab_size}.model" \
      --output_format="piece" \
      < "data/en-de/${file_name}.$language" \
      > "data/en-de/${file_name}.bpe.$language"
  done
done

