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
              y_train, y_mask, y_pos_emb_indices, label_smoothing=True):

    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
    dec_output = self.decoder(enc_output, x_mask, y_train, y_mask,
                              y_pos_emb_indices)
    logits = self.w_logit(dec_output)
    if label_smoothing and (self.hparams.label_smoothing is not None):
      smooth = self.hparams.label_smoothing
      probs = ((1.0 - smooth) * self.softmax(logits) +
               smooth / self.hparams.vocab_size)
      logits = torch.log(probs)

    return logits

  def trainable_parameters(self):
    params = self.parameters()
    return params

  def translate_batch(self, x_train, x_mask, x_pos_emb_indices, beam_size, max_len):
    """

    return:
      all_hyp: [batch_size, n_best] of hypothesis
      all_scores: [batch_size, n_best]
    """
    # (batch_size, src_seq_len, d_model)
    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
    # (batch_size * beam, src_seq_len, d_model)
    enc_output = Variable(enc_output.data.repeat(1, beam_size, 1).view(
                  enc_output.size(0)*beam_size, enc_output.size(1), enc_output.size(2)))
    x_mask = x_mask.repeat(1, beam_size).view(
                  x_mask.size(0)*beam_size, x_mask.size(1))

    batch_size, src_seq_len = x_train.size()
    trg_vocab_size = self.hparams.vocab_size

    beams = [Beam(beam_size, self.hparams) for _ in range(batch_size)]
    beam_to_inst = {beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))}
    n_remain_sents = batch_size
    for i in range(max_len):
      len_dec_seq = i+1 
      # (n_remain_sents * beam, seq_len)
      y_partial = torch.stack([b.get_partial_y() for b in beams if not b.done]).view(-1, len_dec_seq)
      y_partial = Variable(y_partial, volatile=True)
      y_mask = torch.ByteTensor([[0 for _ in range( len_dec_seq)] for _ in range(n_remain_sents*beam_size)])

      y_partial_pos = torch.arange(len_dec_seq).unsqueeze(0)
      # size: (n_remain_sents * beam, seq_len)
      y_partial_pos = y_partial_pos.repeat(n_remain_sents * beam_size, 1)
      y_partial_pos = Variable(torch.FloatTensor(y_partial_pos), volatile=True)

      if self.hparams.cuda:
        y_partial = y_partial.cuda()
        y_partial_pos = y_partial_pos.cuda()
        y_mask = y_mask.cuda()

      dec_output = self.decoder(enc_output, x_mask, y_partial, y_mask, 
                                y_partial_pos)
      # select the dec output for next word
      dec_output = dec_output[:, -1, :]
      logits = self.w_logit(dec_output)
      log_probs = torch.nn.functional.log_softmax(logits, dim=1)
      log_probs =  log_probs.view(n_remain_sents, beam_size, trg_vocab_size)

      active_beam_idx_list = []
      for beam_idx in range(batch_size):
        if beams[beam_idx].done:
          continue
        inst_idx = beam_to_inst[beam_idx]
        if not beams[beam_idx].advance(log_probs.data[inst_idx]):
          active_beam_idx_list += [beam_idx]
      if not active_beam_idx_list: 
        break
      active_inst_idx_list = torch.LongTensor([beam_to_inst[k] for k in active_beam_idx_list])
      beam_to_inst = {beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

      enc_output = select_active_enc_info(enc_output, active_inst_idx_list,
                                          n_remain_sents, src_seq_len, self.hparams.d_model)
      x_mask = select_active_enc_mask(x_mask, active_inst_idx_list, n_remain_sents, src_seq_len)

      n_remain_sents = len(active_inst_idx_list)

    all_hyp, all_scores = [], []
    n_best = 1
    for beam_idx in range(batch_size):
      scores, idxs = torch.sort(beams[beam_idx].scores, dim=0, descending=True)
      all_scores += [scores[:n_best]]

      hyps = [beams[beam_idx].get_y(i) for i in idxs[:n_best]]
      all_hyp += [hyps]
    return all_hyp, all_scores

def select_active_enc_info(enc_output, active_inst_idx_list, n_remain_sents, src_seq_len, d_model):

  original_enc_info_data = enc_output.data.view(n_remain_sents, -1, d_model)
  active_enc_info = original_enc_info_data.index_select(0, active_inst_idx_list)
  active_enc_info = active_enc_info.view(-1, src_seq_len, d_model)
  return Variable(active_enc_info, volatile=True)

def select_active_enc_mask(enc_mask, active_inst_idx_list, n_remain_sents, src_seq_len):
  original_enc_mask = enc_mask.view(n_remain_sents, -1)
  active_enc_mask = original_enc_mask.index_select(0, active_inst_idx_list)
  active_enc_mask = active_enc_mask.view(-1, src_seq_len)
  return active_enc_mask

class Beam(object):

  def __init__(self, size, hparams):
    self.size = size 
    self.done = False
    self.hparams = hparams

    self.tt = torch.cuda if hparams.cuda else torch 

    self.scores = self.tt.FloatTensor(size).zero_()
    self.all_scores = []

    self.prev_ks = []

    self.next_ys = [self.tt.LongTensor(size).fill_(hparams.pad_id)]
    self.next_ys[0][0] = hparams.bos_id 

  def advance(self, word_scores):
    """
    Add word to the beam

    word_scores: (beam_size, trg_vocab_size)
    """
    trg_vocab_size = word_scores.size(1)

    if len(self.prev_ks) > 0:
      beam_score = self.scores.unsqueeze(1).expand_as(word_scores) 
    else:
      # for the first step, all rows in word_scores are the same
      beam_score = word_scores[0]
    flat_beam_score = beam_score.contiguous().view(-1)
    best_scores, best_scores_id = flat_beam_score.topk(self.size, dim=0, 
                                                        largest=True, sorted=True)
    self.all_scores.append(self.scores)
    self.scores = best_scores

    prev_k = best_scores_id / trg_vocab_size
    self.prev_ks.append(prev_k)
    self.next_ys.append(best_scores_id - prev_k * trg_vocab_size)

    if self.next_ys[-1][0] == self.hparams.eos_id:
      self.done = True

  def get_partial_y(self):
    """
    Return the partial y hypothesis for this beam
    """
    if len(self.next_ys) == 1:
      dec_seq = self.next_ys[0].unsqueeze(1)
    else:
      _, keys = torch.sort(self.scores, dim=0, descending=True)
      hyps = [self.get_y(k) for k in keys]
      hyps = [[self.hparams.bos_id] + h for h in hyps]
      dec_seq = torch.from_numpy(np.array(hyps))
    return dec_seq

  def get_y(self, k):
    """
    k: the index in the beam to construct
    """
    hyp = []
    for j in range(len(self.prev_ks)-1, -1, -1):
      hyp.append(self.next_ys[j+1][k])
      k = self.prev_ks[j][k]
    return hyp[::-1]

    return hyp[::-1]
