

PUNC_NORM=data_scripts/scripts/tokenizer/normalize-punctuation.perl
TOKENIZER=data_scripts/scripts/tokenizer/tokenizer.perl
CLEAN=data_scripts/scripts/training/clean-corpus-n.perl
TRUECASE_TRAIN=data_scripts/scripts/recaser/train-truecaser.perl
TRUECASE_APPLY=data_scripts/scripts/recaser/truecase.perl
BPE_TRAIN=spm_encode
BPE_APPLY=spm_decode
MAKE_DICT=tmp

RAW_DIR=raw/en-de/
OUT_DIR=preproc/en-de/

output_path=clean_piece_24k_shared/en-de
vocab_size=24000
mkdir -p ${output_path}
#mkdir $OUT_DIR

# puctuation normalization
#for file in train dev2010 tst2014 
#  do 
#  for lan in en de
#    do
#      $PUNC_NORM -l $lan < $RAW_DIR$file.$lan > $OUT_DIR$file.norm.$lan
#    done
#  done

# tokenizer
#for file in train dev2010 tst2014
#  do
#  for lan in en de
#    do
#      $TOKENIZER -l $lan < $OUT_DIR$file.norm.$lan > $OUT_DIR$file.tok.$lan
#    done
#  done

# clean corpus; only clean training corpus
#$CLEAN $OUT_DIR"train.tok" en de $OUT_DIR"train.clean" 1 80


# train truecaser
#$TRUECASE_TRAIN -corpus $OUT_DIR"train.clean.en" -model $OUT_DIR"truecase.model.en"
#$TRUECASE_TRAIN -corpus $OUT_DIR"train.clean.de" -model $OUT_DIR"truecase.model.de"

# apply truecaser
# training data
#for lan in en de
#  do
#    $TRUECASE_APPLY -model $OUT_DIR"truecase.model."$lan < $OUT_DIR"train.clean."$lan > $OUT_DIR"train.truecase."$lan
#  done

# dev test data
#for file in dev2010 tst2014
#  do
#  for lan in en de
#    do
#      $TRUECASE_APPLY -model $OUT_DIR"truecase.model."$lan < $OUT_DIR$file.tok.$lan > $OUT_DIR$file.truecase.$lan
#    done
#  done

# train bpe
spm_train \
  --input="${OUT_DIR}train.truecase.en,${OUT_DIR}train.truecase.de" \
  --model_prefix="${output_path}/shared_${vocab_size}" \
  --vocab_size=${vocab_size} \
  --model_type="unigram"

# apply bpe
for file in train dev2010 tst2014
  do
  for lan in en de
    do
      spm_encode \
        --extra_options="eos" \
        --model="${output_path}/shared_${vocab_size}.model" \
        --output_format="piece" \
        < $OUT_DIR$file'.truecase.'$lan \
        > "${output_path}/$file.$lan"
    done
  done

# make dictionary
