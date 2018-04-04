#!/bin/bash

vocab_size=5000

src="en"
tgt="zh"
languages=(${src} ${tgt})
file_names=(
  "train"
  "dev2010"
)

mkdir -p "data/bpe_${vocab_size}/${src}-${tgt}"

for language in ${languages[@]}
do
  echo "Training BPE on train.$language"
  spm_train \
    --input="data/raw/${src}-${tgt}/train.$language" \
    --model_prefix="data/bpe_${vocab_size}/${src}-${tgt}/$language.bpe.${vocab_size}" \
    --vocab_size=${vocab_size} \
    --model_type="bpe"

  for file_name in ${file_names[@]}
  do
    echo "Generating BPE for ${file_name}.$language"
    spm_encode \
      --extra_options="eos" \
      --model="data/bpe_${vocab_size}/${src}-${tgt}/$language.bpe.${vocab_size}.model" \
      --output_format="piece" \
      < "data/raw/${src}-${tgt}/${file_name}.$language" \
      > "data/bpe_${vocab_size}/${src}-${tgt}/${file_name}.bpe.$language"
  done
done

