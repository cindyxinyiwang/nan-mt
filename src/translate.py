

import argparse
import _pickle as pickle
import shutil
import gc
import os
import sys
import time

import numpy as np

from data_utils import DataLoader
from hparams import *
from utils import *
from models import *

import torch
import torch.nn as nn
from torch.autograd import Variable

class TranslationHparams(exp4_v1):
  dataset = "IWSLT 2016 En-De with BPE 32K Shared Vocab"
  data_path = "data/bpe_32k_shared_vocab/en-de"

  source_train = "tiny.en"
  target_train = "tiny.de"
  source_valid = "tiny.en"
  target_valid = "tiny.de"
  source_test = "tiny.en"
  target_test = "tiny.de"

  source_vocab = "shared_32000.vocab"
  target_vocab = "shared_32000.vocab"

  unk = "<unk>"
  bos = "<s>"
  eos = "</s>"
  unk_id = 31997
  eos_id = 31998
  bos_id = 31999

  pad = bos
  pad_id = bos_id

  cuda = True

  beam_size = 2
  max_len = 100
  batch_size = 10
  out_file = "outputs/trans.test"
  merge_bpe = True
  filtered_tokens = set([eos_id, bos_id])

parser = argparse.ArgumentParser(description="Neural MT translator")

add_argument(parser, "model_dir", type="str", default="outputs",
             help="root directory of saved model")
# add_argument(parser, "src_file", type="str", default="data/src",
#              help="src file to translate")
# add_argument(parser, "ref_file", type="str", default="data/trg",
#              help="reference file")
# add_argument(parser, "out_dir", type="str", default="outputs",
#              help="output dir of translation")
# add_argument(parser, "beam_size", type="int", default="5",
#              help="beam size")
# add_argument(parser, "max_len", type="int", default="50",
#              help="max len")

args = parser.parse_args()

model_file_name = os.path.join(args.model_dir, "model.pt")
model = torch.load(model_file_name)

hparams = TranslationHparams()
hparams.cuda = True
model.hparams.cuda = hparams.cuda

data = DataLoader(hparams=hparams, decode=True)

out_file = open(hparams.out_file, 'w')
end_of_epoch = False
num_sentences = 0
while not end_of_epoch:
  ((x_test, x_mask, x_pos_emb_indices, x_count),
   (y_test, y_mask, y_pos_emb_indices, y_count),
   end_of_epoch) = data.next_test(test_batch_size=hparams.batch_size)
  num_sentences += x_test.size(0)

  all_hyps, all_scores = model.translate_batch(x_test, x_mask, x_pos_emb_indices,
 	                                            hparams.beam_size, hparams.max_len)
  for h in all_hyps:
    h_best = h[0]
    h_best_words = map(lambda wi: data.target_index_to_word[wi],
                       filter(lambda wi: wi not in hparams.filtered_tokens, h_best))
    line = ' '.join(h_best_words)
    if hparams.merge_bpe:
        line = line.replace(' @@', '')
    out_file.write(line + '\n')
    out_file.flush()
  
  print("Translated {0} sentences".format(num_sentences))
  break

out_file.close()

