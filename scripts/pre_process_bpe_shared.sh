#!/bin/bash

vocab_size=28000

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

output_path="data/bpe_28k_shared/en-de"

mkdir -p ${output_path}

echo "Training BPE on train.{en,de}"
spm_train \
  --input="data/raw/en-de/train.en,data/raw/en-de/train.de" \
  --character_coverage=0.9999 \
  --model_prefix="${output_path}/shared_${vocab_size}" \
  --vocab_size=${vocab_size} \
  --model_type="bpe"

for language in ${languages[@]}
do
  for file_name in ${file_names[@]}
  do
    echo "Generating BPE for ${file_name}.$language"
    spm_encode \
      --extra_options="eos" \
      --model="${output_path}/shared_${vocab_size}.model" \
      --output_format="piece" \
      < "data/raw/en-de/${file_name}.$language" \
      > "${output_path}/${file_name}.$language"
  done
done

