from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


class Iwslt16EnDeBpe32SharedParams(object):
  """Basic params for the data set. Set other hparams via inheritance."""

  dataset = "IWSLT 2016 En-De with BPE 32K Shared Vocab"
  data_path = "data/bpe_32k_shared_vocab/en-de"

  cuda = True
  train_limit = None
  source_train = "train.en"
  target_train = "train.de"
  source_valid = "dev2010.en"
  target_valid = "dev2010.de"

  source_vocab = "shared_32000.vocab"
  target_vocab = "shared_32000.vocab"

  source_test = "tst2014.en"
  target_test = "tst2014.de"

  vocab_size = 32000 + 1
  max_train_len = 625  # None
  max_len = 625

  unk = "<unk>"
  bos = "<s>"
  eos = "</s>"
  pad = "<pad>"

  unk_id = 31997
  eos_id = 31998
  bos_id = 31999
  pad_id = 32000

  fixed_lr = None


class exp6_v1(Iwslt16EnDeBpe32SharedParams):
  d_word_vec = 288  # size of word and positional embeddings
  d_model = 288  # size of hidden states
  d_inner = 512  # hidden dimension of the position-wise ff
  d_k = 64  # dim of attn keys
  d_v = 64  # dim of attn values

  n_layers = 5  # number of layers in a Transformer stack
  n_heads = 2   # number of attention heads

  dropout = 0.1  # probability of dropping

  share_emb_and_softmax = True  # share embedding and softmax
  share_source_and_target_emb = False  # share source and target embeddings

  # training
  batch_size = 32
  fixed_lr = 1e-4
  label_smoothing = 0.1

  n_epochs = 50000
  n_train_steps = 100000
  n_warm_ups = 750

# Put all Hparams in a dictionary
H_PARAMS_DICT = {
  "exp6_v1": exp6_v1,
}

