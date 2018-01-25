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
  def __init__(self, n_src_vocab, hparams, n_layers=6, n_head=8, d_k=64, d_v=64,
                d_word_vec=512, dim=512, d_inner=1024, dropout=0.1, *args, **kwargs):
    #raise NotImplementedError("Bite me!")
    super(Encoder, self).__init__()
    self.n_max_seq = hparams.max_len
    self.dim = dim

    self.pos_emb = PositionalEmbedding(hparams)
    self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=hparams.pad_id)

    self.layer_stack = nn.ModuleList([EncoderLayer(dim, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

  def forward(self, x_train, x_mask, x_pos_emb_indices, attn_mask=None, *args, **kwargs):
    """Performs a forward pass.

    Args:
      source_indices: Torch Tensor of size [batch_size, max_len]
      source_lengths: position encoding
    """

    print("HERE 1")
    enc_input = self.src_word_emb(x_train)
    print("HERE 2")
    enc_input += self.pos_emb(x_pos_emb_indices, x_mask)

    # TODO(hyhieu): continue here
    sys.exit(0)

    enc_output = enc_input
    for enc_layer in self.layer_stack:
      enc_output = enc_layer(enc_output, attn_mask=attn_mask)
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
  def __init__(self, n_src_vocab, n_trg_vocab, hparams, n_layers=6, n_head=8,
                d_word_vec=512, dim=512, d_inner=1024, d_k=64, d_v=64, 
                dropout=0.1, *args, **kwargs):
    super(Transformer, self).__init__()
    self.encoder = Encoder(n_src_vocab, hparams, n_layers, n_head, d_word_vec=d_word_vec,
                            dim=dim, d_inner=d_inner, dropout=dropout)
    self.dropout = nn.Dropout(dropout)
    assert dim == d_word_vec

  def forward(self, x_train, x_mask, x_pos_emb_indices,
              y_train, y_mask, y_pos_emb_indices):

    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)


