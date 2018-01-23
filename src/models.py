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

    self.position_enc = nn.Embedding(self.n_max_seq+1, d_word_vec, padding_idx=hparams.pad_id)  # index 0 as padding pos
    # init sinusoid position encoding
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
                            if pos != 0 else np.zeros(d_word_vec) for pos in range(self.n_max_seq+1)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    self.position_enc.weight.data = torch.from_numpy(position_enc).type(torch.FloatTensor)

    self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=hparams.pad_id)

    self.layer_stack = nn.ModuleList([EncoderLayer(dim, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

  def forward(self, source_indices, source_lengths, *args, **kwargs):
    """Performs a forward pass.

    Args:
      source_indices: Torch Tensor of size [batch_size, max_len]
      source_lengths: position encoding
    """
    #raise NotImplementedError("Bite me!")
    #print(len(source_indices), len(source_lengths))
    #print(source_lengths)
    #source_pos = [ [i for i in range(l)] for l in source_lengths]
    enc_input = self.src_word_emb(source_indices)
    enc_input += self.position_enc(source_lengths)

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
  def __init__(self, n_src_vocab, n_trg_vocab, hparams, n_layers=6, n_head=8,
                d_word_vec=512, dim=512, d_inner=1024, d_k=64, d_v=64, 
                dropout=0.1, *args, **kwargs):
    super(Transformer, self).__init__()
    self.encoder = Encoder(n_src_vocab, hparams, n_layers, n_head, d_word_vec=d_word_vec,
                            dim=dim, d_inner=d_inner, dropout=dropout)
    self.dropout = nn.Dropout(dropout)
    assert dim == d_word_vec

  def forward(self, src_seq, src_len, trg_seq, trg_len):

    enc_output = self.encoder(src_seq, src_len)



