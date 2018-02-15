from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
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
      [EncoderLayer(self.hparams.d_model,
                    self.hparams.d_inner,
                    self.hparams.n_heads,
                    self.hparams.d_k,
                    self.hparams.d_v,
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
      x_mask: Torch Tensor of size [batch_size, max_len]. 1 means to ignore a
        position.
      x_pos_emb_indices: used to compute positional embeddings.

    Returns:
      enc_output: Tensor of size [batch_size, max_len, d_model].
    """

    batch_size, max_len = x_train.size()

    # [batch_size, max_len, 1] -> [batch_size, max_len, d_word_vec]
    pos_emb_mask = x_mask.clone().unsqueeze(2).expand(
      -1, -1, self.hparams.d_word_vec)
    pos_emb = self.pos_emb(x_pos_emb_indices, pos_emb_mask)

    # [batch_size, max_len, d_word_vec]
    word_emb = self.word_emb(x_train)
    enc_input = word_emb + pos_emb

    # [batch_size, 1, max_len] -> [batch_size, len_q, len_k]
    attn_mask = x_mask.clone().unsqueeze(1).expand(-1, max_len, -1)
    enc_output = enc_input
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
      y_mask: Torch Tensor of size [batch_size, max_len]. 1 means to ignore a
        position.
      y_pos_emb_indices: used to compute positional embeddings.

    Returns:
      y_states: tensor of size [batch_size, max_len, d_model], the highest
        output layer.
    """

    batch_size, x_len = x_mask.size()
    batch_size, y_len = y_mask.size()

    # [batch_size, y_len, 1] -> [batch_size, y_len, d_word_vec]
    pos_emb_mask = y_mask.clone().unsqueeze(2).expand(
      -1, -1, self.hparams.d_word_vec)
    pos_emb = self.pos_emb(y_pos_emb_indices, pos_emb_mask)

    # [batch_size, x_len, d_word_vec]
    word_emb = self.word_emb(y_train)
    dec_input = word_emb + pos_emb

    # [batch_size, 1, y_len] -> [batch_size, y_len, y_len]
    y_attn_mask = (y_mask.clone().unsqueeze(1).expand(-1, y_len, -1) +
                   get_attn_subsequent_mask(y_train,
                                            pad_id=self.hparams.pad_id))
    y_attn_mask = torch.gt(y_attn_mask, 0)

    # [batch_size, 1, x_len] -> [batch_size, y_len, x_len]
    x_attn_mask = x_mask.clone().unsqueeze(1).expand(-1, y_len, -1)

    dec_output = dec_input
    for dec_layer in self.layer_stack:
      dec_output = dec_layer(dec_output, x_states,
                             self_attn_mask=y_attn_mask,
                             dec_enc_attn_mask=x_attn_mask)
    return dec_output

class Transformer(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    super(Transformer, self).__init__()

    self.hparams = hparams
    self.encoder = Encoder(hparams)
    self.dropout = nn.Dropout(hparams.dropout)
    self.decoder = Decoder(hparams)
    self.w_logit = nn.Linear(hparams.d_model, hparams.vocab_size, bias=False)
    if hparams.share_emb_and_softmax:
      self.w_logit.weight = self.decoder.word_emb.weight

    init.xavier_normal(self.w_logit.weight)

    if hparams.label_smoothing is not None:
      self.softmax = nn.Softmax(dim=-1)
      smooth = np.full([1, 1, hparams.vocab_size], 1 / hparams.vocab_size,
                       dtype=np.float32)
      self.smooth = torch.FloatTensor(smooth)
      if self.hparams.cuda:
        self.smooth = self.smooth.cuda()

  def forward(self, x_train, x_mask, x_pos_emb_indices,
              y_train, y_mask, y_pos_emb_indices):

    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
    dec_output = self.decoder(enc_output, x_mask, y_train, y_mask,
                              y_pos_emb_indices)
    logits = self.w_logit(dec_output)
    if self.hparams.label_smoothing is not None:
      smooth = self.hparams.label_smoothing
      probs = ((1.0 - smooth) * self.softmax(logits) +
               smooth / self.hparams.vocab_size)
      logits = torch.log(probs)

    return logits

  def trainable_parameters(self):
    params = self.parameters()
    return params
