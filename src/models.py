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
from utils import *

class Encoder(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    super(Encoder, self).__init__()

    self.hparams = hparams
    assert self.hparams.d_word_vec == self.hparams.d_model

    self.pos_emb = PositionalEmbedding(hparams)
    self.word_emb = nn.Embedding(self.hparams.vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)

    self.layer_stack = nn.ModuleList(
      [EncoderLayer(self.hparams.d_word_vec, self.hparams.d_inner,
                    self.hparams.n_heads, self.hparams.d_k, self.hparams.d_v,
                    dropout=self.hparams.dropout)
       for _ in range(self.hparams.n_layers)])

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
      enc_output: Tensor of size [batch_size, max_len, d_model].
    """

    # TODO(cindyxinyiwang): handle x_mask
    word_emb = self.word_emb(x_train)
    pos_emb = self.pos_emb(x_pos_emb_indices, x_mask)
    enc_input = word_emb + pos_emb

    enc_output = enc_input
    # create attn_mask. For encoder, q and k are both encoder states
    # make it to (batch_size, len_q, len_k)
    attn_mask = get_attn_padding_mask(x_train, x_train)
    for enc_layer in self.layer_stack:
      enc_output = enc_layer(enc_output, attn_mask=attn_mask)

    return enc_output

class Decoder(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    """Store hparams and creates modules."""

    super(Decoder, self).__init__()
    self.hparams = hparams

    self.pos_emb = PositionalEmbedding(self.hparams)
    self.word_emb = nn.Embedding(self.hparams.vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)

    self.layer_stack = nn.ModuleList(
      [DecoderLayer(self.hparams.d_model, self.hparams.d_inner,
                    self.hparams.n_heads, self.hparams.d_k, self.hparams.d_v,
                    dropout=self.hparams.dropout)
       for _ in range(self.hparams.n_layers)])

    if self.hparams.cuda:
      self.word_emb = self.word_emb.cuda()
      self.pos_emb = self.pos_emb.cuda()
      self.layer_stack = self.layer_stack.cuda()

  def forward(self, x_states, x_mask, y_train, y_mask, y_pos_emb_indices):
    """Performs a forward pass.

    Args:
      x_states: tensor of size [batch_size, max_len, d_model], input
        attention memory.
      x_mask: tensor of size [batch_size, max_len]. input mask.
      y_train: Torch Tensor of size [batch_size, max_len]
      y_mask: Torch Tensor of size [batch_size, max_len]. 0 means to ignore a
        position, 1 means to keep the position.
      y_pos_emb_indices: used to compute positional embeddings.

    Returns:
      y_states: tensor of size [batch_size, max_len, d_model], the highest
        output layer.
    """

    print("-" * 80)
    print("HERE. Calling Decoder")

    word_emb = self.word_emb(y_train)
    pos_emb = self.pos_emb(y_pos_emb_indices, y_mask)
    dec_input = word_emb + pos_emb

    # TODO(hyhieu): check the masks!!!
    dec_output = dec_input
    for dec_layer in self.layer_stack:
      dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
        dec_output, x_states, slf_attn_mask=y_mask, dec_enc_attn_mask=x_mask)

    sys.exit(0)

class Transformer(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    super(Transformer, self).__init__()

    self.hparams = hparams
    self.encoder = Encoder(hparams)
    self.dropout = nn.Dropout(hparams.dropout)
    self.decoder = Decoder(hparams)

  def forward(self, x_train, x_mask, x_pos_emb_indices,
              y_train, y_mask, y_pos_emb_indices):

    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
    dec_output = self.decoder(enc_output, x_mask, y_train, y_mask,
                              y_pos_emb_indices)

    return dec_output
