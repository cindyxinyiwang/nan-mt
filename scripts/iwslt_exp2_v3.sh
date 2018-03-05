#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH -t 0


module load singularity
singularity shell --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img


PYTHONIOENCODING=utf-8:surrogateescape python3 src/main.py \
  --output_dir="outputs_exp2_v3_newnorm" \
  --log_every=100 \
  --eval_every=1000 \
  --reset_output_dir \
  --no_load_model \
  --data_path="data/bpe_28k_shared/en-de/" \
  --source_train="train.en" \
  --target_train="train.de" \
  --source_valid="dev2010.en" \
  --target_valid="dev2010.de" \
  --source_vocab="shared_28000.vocab" \
  --target_vocab="shared_28000.vocab" \
  --source_test="tst2014.en" \
  --target_test="tst2014.de" \
  --batch_size=32 \
  --n_train_sents=250000 \
  --max_len=600 \
  --d_word_vec=256 \
  --d_model=256 \
  --d_inner=512 \
  --n_layers=5 \
  --d_k=128 \
  --d_v=128 \
  --n_heads=2 \
  --n_warm_ups=750 \
  --dropout=0.1 \
  --cuda \
  --n_train_steps 150000 \
  "$@"

