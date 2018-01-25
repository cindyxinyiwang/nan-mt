from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle
import shutil
import os
import sys
import time

import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
import numpy as np
from layers import *


class Encoder(nn.Module):
  def __init__(self, hparams, n_layers=6, n_head=8, d_k=64, d_v=64,
               d_inner=1024, dropout=0.1, *args, **kwargs):
    super(Encoder, self).__init__()

    self.hparams = hparams

    self.pos_emb = PositionalEmbedding(hparams)
    self.word_emb = nn.Embedding(self.hparams.vocab_size,
                                 self.hparams.embedding_size,
                                 padding_idx=hparams.pad_id)

    self.layer_stack = nn.ModuleList(
      [EncoderLayer(self.hparams.embedding_size, d_inner, n_head, d_k, d_v,
                    dropout=dropout)
       for _ in range(n_layers)])

    if self.hparams.cuda:
      self.word_emb = self.word_emb.cuda()
      self.pos_emb = self.pos_emb.cuda()
      self.layer_stack = self.layer_stack.cuda()

  def forward(self, x_train, x_mask, x_pos_emb_indices):
    """Performs a forward pass.

    Args:
      x_train: Torch Tensor of size [batch_size, max_len]
      x_mask: Torch Tensor of size [batch_size, max_len]. 0 means to ignore a
        position, 1 means to keep the position.
      x_pos_emb_indices: used to compute positional embeddings.

    Returns:
      enc_output: Tensor of size [batch_size, max_len, hidden_size].
    """

    # TODO(cindyxinyiwang): handle x_mask
    word_emb = self.word_emb(x_train)
    pos_emb = self.pos_emb(x_pos_emb_indices, x_mask)
    enc_input = word_emb + pos_emb

    enc_output = enc_input
    for enc_layer in self.layer_stack:
      enc_output = enc_layer(enc_output)

    return enc_output

class Decoder(nn.Module):
  def __init__(self, *args, **kwargs):
    raise NotImplementedError("Bite me!")

  def forward(self, source_states):
    """Performs a forward pass.

    Args:
      source_states: Torch Tensor of size [batch_size, max_len]
    """
    raise NotImplementedError("Bite me!")

class Transformer(nn.Module):
  def __init__(self, hparams, n_layers=6, n_head=8, d_inner=1024, d_k=64,
               d_v=64, dropout=0.1, *args, **kwargs):
    super(Transformer, self).__init__()
    self.encoder = Encoder(hparams, n_layers, n_head,
                           d_inner=d_inner, dropout=dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x_train, x_mask, x_pos_emb_indices,
              y_train, y_mask, y_pos_emb_indices):

    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)


