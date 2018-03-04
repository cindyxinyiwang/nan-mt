from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

class TranslationHparams(Iwslt16EnDeBpe32SharedParams):
  dataset = "IWSLT 2016 En-De with BPE 32K Shared Vocab"

parser = argparse.ArgumentParser(description="Neural MT translator")

add_argument(parser, "cuda", type="bool", default=False, help="GPU or not")
add_argument(parser, "data_path", type="str", default=None, help="path to all data")
add_argument(parser, "model_dir", type="str", default="outputs", help="root directory of saved model")
add_argument(parser, "source_test", type="str", default=None, help="name of source test file")
add_argument(parser, "target_test", type="str", default=None, help="name of target test file")
add_argument(parser, "beam_size", type="int", default=None, help="beam size")
add_argument(parser, "max_len", type="int", default=300, help="maximum len considered on the target side")
add_argument(parser, "batch_size", type="int", default=32, help="")
add_argument(parser, "merge_bpe", type="bool", default=True, help="")
add_argument(parser, "source_vocab", type="str", default=None, help="name of source vocab file")
add_argument(parser, "target_vocab", type="str", default=None, help="name of target vocab file")
add_argument(parser, "n_train_sents", type="int", default=None, help="max number of training sentences to load")


args = parser.parse_args()

model_file_name = os.path.join(args.model_dir, "model.pt")
model = torch.load(model_file_name)
model.eval()

out_file = os.path.join(args.model_dir, "trans")

hparams = TranslationHparams(
  data_path=args.data_path,
  source_vocab=args.source_vocab,
  target_vocab=args.target_vocab,
  source_test = args.source_test,
  target_test = args.target_test,
  cuda=args.cuda,
  beam_size=args.beam_size,
  max_len=args.max_len,
  batch_size=args.batch_size,
  n_train_sents=args.n_train_sents,
  merge_bpe=args.merge_bpe,
  out_file=out_file,
)

hparams.add_param("filtered_tokens", set([hparams.pad_id, hparams.eos_id, hparams.bos_id]))

hparams.cuda = True
model.hparams.cuda = hparams.cuda

data = DataLoader(hparams=hparams, decode=True)

out_file = open(hparams.out_file, 'w')
end_of_epoch = False
num_sentences = 0
while not end_of_epoch:
  ((x_test, x_mask, x_pos_emb_indices, x_count),
   (y_test, y_mask, y_pos_emb_indices, y_count),
   batch_size, end_of_epoch) = data.next_test(test_batch_size=hparams.batch_size)

  ## DEBUG
  # print("x_test", x_test)
  # print("y_test", y_test)
  # y = y_test.cpu().data.numpy()
  # for _y in y:
  #   log_string = ""
  #   for __y in _y:
  #     log_string += "{0:<10} ".format(data.source_index_to_word[__y])
  #   print(log_string)
  ## END OF DEBUG

  num_sentences += batch_size

  # The normal, correct way:
  all_hyps, all_scores = model.translate_batch(
    x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len)

  # For debugging:
  # model.debug_translate_batch(
  #   x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len,
  #   y_test, y_mask, y_pos_emb_indices)
  # sys.exit(0)

  for h in all_hyps:
    h_best = h[0]
    h_best_words = map(lambda wi: data.target_index_to_word[wi],
                       filter(lambda wi: wi not in hparams.filtered_tokens, h_best))
    if hparams.merge_bpe:
      line = ''.join(h_best_words)
      line = line.replace('▁', ' ')
    else:
      line = ' '.join(h_best_words)
    line = line.strip()
    out_file.write(line + '\n')
    out_file.flush()
  
  print("Translated {0} sentences".format(num_sentences))

out_file.close()

