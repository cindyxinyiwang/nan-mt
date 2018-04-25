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

add_argument(parser, "cuda", type="bool", default=False,
             help="GPU or not")
add_argument(parser, "data_path", type="str", default=None,
             help="path to all data")
add_argument(parser, "model_dir", type="str", default="outputs",
             help="root directory of saved model")
add_argument(parser, "source_test", type="str", default=None,
             help="name of source test file")
add_argument(parser, "target_test", type="str", default=None,
             help="name of target test file")
add_argument(parser, "beam_size", type="int", default=None,
             help="beam size")
add_argument(parser, "max_len", type="int", default=300,
             help="maximum len considered on the target side")
add_argument(parser, "non_batch_translate", type="bool", default=False,
             help="use non-batched translation")
add_argument(parser, "batch_size", type="int", default=32, help="")
add_argument(parser, "merge_bpe", type="bool", default=True, help="")
add_argument(parser, "source_vocab", type="str", default=None,
             help="name of source vocab file")
add_argument(parser, "target_vocab", type="str", default=None,
             help="name of target vocab file")
add_argument(parser, "n_train_sents", type="int", default=None,
             help="max number of training sentences to load")
add_argument(parser, "out_file", type="str", default="trans",
             help="output file for hypothesis")
add_argument(parser, "raml_src_tau", type="float", default=1.0,
             help="Temperature parameter of RAML")
add_argument(parser, "raml_source", type="bool", default=False,
             help="Sample a corrupted source sentence")
add_argument(parser, "n_corrupts", type="int", default=0,
             help="Number of source corruptions")
add_argument(parser, "n_cleans", type="int", default=0,
             help="Number of cleaned sentences in source corruptions")
add_argument(parser, "src_pad_corrupt", type="bool", default=False,              
             help="Use pad id as token for corrupted source sentence") 
add_argument(parser, "dist_corrupt", type="bool", default=False,
             help="Use distance of word vector to select corrupted words")
add_argument(parser, "dist_corrupt_tau", type="float", default=20.0,
             help="Temperature parameter of corrupting by distance")
add_argument(parser, "glove_emb_file", type="str", default=None,
             help="Path to the word embedding file for raml corruption")
add_argument(parser, "glove_emb_dim", type="int", default=None,
             help="word embed dimension of the glove emb")
add_argument(parser, "max_glove_vocab_size", type="int", default=None,
             help="maximum number of glove vocab to load")

args = parser.parse_args()

model_file_name = os.path.join(args.model_dir, "model.pt")
if not args.cuda:
  model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
else:
  model = torch.load(model_file_name)
model.eval()

out_file = os.path.join(args.model_dir, args.out_file)

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
  raml_source=args.raml_source,
  raml_src_tau=args.raml_src_tau,
  n_corrupts=args.n_corrupts,
  n_cleans=args.n_cleans,
  dist_corrupt=args.dist_corrupt,
  dist_corrupt_tau=args.dist_corrupt_tau,
  glove_emb_file=args.glove_emb_file,
  glove_emb_dim=args.glove_emb_dim,
  max_glove_vocab_size=args.max_glove_vocab_size,
  src_pad_corrupt=args.src_pad_corrupt,
)

data = DataLoader(hparams=hparams, decode=True)
hparams.add_param("source_vocab_size", data.source_vocab_size)
hparams.add_param("target_vocab_size", data.target_vocab_size)
hparams.add_param("pad_id", data.pad_id)
hparams.add_param("unk_id", data.unk_id)
hparams.add_param("bos_id", data.bos_id)
hparams.add_param("eos_id", data.eos_id)
hparams.add_param(
  "filtered_tokens",
  set([model.hparams.pad_id, model.hparams.eos_id, model.hparams.bos_id]))
model.hparams.cuda = hparams.cuda

out_file = open(hparams.out_file, 'w', encoding='utf-8')
end_of_epoch = False
num_sentences = 0
while not end_of_epoch:
  if hparams.raml_source:
    ((x_test_raml, x_test, x_mask, x_pos_emb_indices, x_count),
     (y_test, y_mask, y_pos_emb_indices, y_count),
     batch_size, end_of_epoch) = data.next_test(
       test_batch_size=hparams.batch_size, raml=True)
  else:
    ((x_test, x_mask, x_pos_emb_indices, x_count),
     (y_test, y_mask, y_pos_emb_indices, y_count),
     batch_size, end_of_epoch) = data.next_test(
       test_batch_size=hparams.batch_size)

  num_sentences += batch_size

  # The normal, correct way:
  if args.non_batch_translate:
    #print("non batched translate...")
    all_hyps, all_scores = model.translate(
      x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len)
  else:
    #print("batched translate...")
    if hparams.raml_source:
      #all_hyps, all_scores = model.translate_batch_corrupt(
      #  x_test_raml, x_mask, x_pos_emb_indices, hparams.beam_size,
      #  hparams.max_len, raml_source=args.raml_source, n_corrupts=args.n_corrupts)
      all_hyps, all_scores = model.translate_batch(
        x_test_raml, x_mask, x_pos_emb_indices, hparams.beam_size,
        hparams.max_len, raml_source=args.raml_source, n_corrupts=args.n_corrupts)
    else:
      all_hyps, all_scores = model.translate_batch(
        x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len)

  # For debugging:
  # model.debug_translate_batch(
  #   x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len,
  #   y_test, y_mask, y_pos_emb_indices)
  # sys.exit(0)

  for h in all_hyps:
    #print(h)
    h_best = h[0]
    h_best_words = map(
      lambda wi: data.target_index_to_word[wi],
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
  sys.stdout.flush()

out_file.close()

