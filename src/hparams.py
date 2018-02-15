from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


class Iwslt16EnDeBpe32Params(object):
  """For small experiments."""

  dataset = "IWSLT 2016 En-De with BPE 16K"
  data_path = "data/bpe_32k/en-de"

  train_limit = None
  source_train = "train.bpe.en"
  target_train = "train.bpe.de"

  source_valid = "dev2010.bpe.en"
  target_valid = "dev2010.bpe.de"

  source_vocab = "en.bpe.16000.vocab"
  target_vocab = "de.bpe.16000.vocab"

  vocab_size = 32000
  max_train_len = 200

  unk = "<unk>"
  bos = "<s>"
  eos = "</s>"
  unk_id = 15997
  eos_id = 15998
  bos_id = 15999

  cuda = True
  batch_size = 32
  num_epochs = 50


class Iwslt16EnDeBpe16Params(object):
  """For small experiments."""

  dataset = "IWSLT 2016 En-De with BPE 16K"
  data_path = "data/bpe_16k/en-de"

  train_limit = None
  source_train = "train.bpe.en"
  target_train = "train.bpe.de"

  source_valid = "dev2010.bpe.en"
  target_valid = "dev2010.bpe.de"

  source_vocab = "en.bpe.16000.vocab"
  target_vocab = "de.bpe.16000.vocab"

  vocab_size = 16000
  max_train_len = 200

  unk = "<unk>"
  bos = "<s>"
  eos = "</s>"
  unk_id = 15997
  eos_id = 15998
  bos_id = 15999

  pad = bos
  pad_id = bos_id

  batch_size = 192
  num_epochs = 50
  cuda = True

  d_word_vec = 256  # size of word and positional embeddings
  d_model = 256  # size of hidden states
  d_inner = 512  # hidden dimension of the position-wise ff
  d_k = 64  # dimension of attention keys
  d_v = 64  # dimension of attention values

  n_layers = 5  # number of layers in a Transformer stack
  n_heads = 2   # number of attention heads

  dropout = 0.1  # probability of dropping

  share_emb_and_softmax = True  # share embedding and softmax
  learning_rate = 1.0 / np.sqrt(d_model)


class Iwslt16EnDeTinyParams(Iwslt16EnDeBpe16Params):
  """Shrinks Iwslt16EnDeBpe16Params for sanity check."""

  dataset = "Tiny IWSLT 2016 En-De with BPE 16K"
  train_limit = 50000

  batch_size = 128
  num_epochs = 10
  cuda = True

  d_word_vec = 256
  d_model = 256
  d_inner = 128

  d_k = 64
  d_v = 64

  n_layers = 5
  n_heads = 2


