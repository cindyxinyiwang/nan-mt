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
    self.word_emb = nn.Embedding(self.hparams.source_vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)
    self.emb_scale = np.sqrt(self.hparams.d_model)

    self.layer_stack = nn.ModuleList(
      [EncoderLayer(hparams) for _ in range(self.hparams.n_layers)])

    self.dropout = nn.Dropout(self.hparams.dropout)
    if self.hparams.cuda:
      self.word_emb = self.word_emb.cuda()
      self.pos_emb = self.pos_emb.cuda()
      self.layer_stack = self.layer_stack.cuda()
      self.dropout = self.dropout.cuda()

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

    # [batch_size, max_len, d_word_vec]
    pos_emb = self.pos_emb(x_train)
    word_emb = self.word_emb(x_train) * self.emb_scale
    enc_input = word_emb + pos_emb

    # [batch_size, 1, max_len] -> [batch_size, len_q, len_k]
    attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1).contiguous()
    enc_output = self.dropout(enc_input)
    for enc_layer in self.layer_stack:
      enc_output = enc_layer(enc_output, attn_mask=attn_mask)

    return enc_output

class Decoder(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    """Store hparams and creates modules."""
    super(Decoder, self).__init__()
    self.hparams = hparams

    self.pos_emb = PositionalEmbedding(hparams)
    self.word_emb = nn.Embedding(self.hparams.target_vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)
    self.emb_scale = np.sqrt(self.hparams.d_model)

    self.layer_stack = nn.ModuleList(
      [DecoderLayer(hparams) for _ in range(self.hparams.n_layers)])

    self.dropout = nn.Dropout(self.hparams.dropout)

    if self.hparams.cuda:
      self.word_emb = self.word_emb.cuda()
      self.pos_emb = self.pos_emb.cuda()
      self.layer_stack = self.layer_stack.cuda()
      self.dropout = self.dropout.cuda()

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

    # [batch_size, x_len, d_word_vec]
    pos_emb = self.pos_emb(y_train)
    word_emb = self.word_emb(y_train) * self.emb_scale
    dec_input = word_emb + pos_emb

    # [batch_size, 1, y_len] -> [batch_size, y_len, y_len]
    y_time_mask = get_attn_subsequent_mask(y_train, pad_id=self.hparams.pad_id)
    y_attn_mask = y_mask.unsqueeze(1).expand(-1, y_len, -1).contiguous()
    y_attn_mask = y_attn_mask | y_time_mask

    # [batch_size, 1, x_len] -> [batch_size, y_len, x_len]
    x_attn_mask = x_mask.unsqueeze(1).expand(-1, y_len, -1).contiguous()

    dec_output = self.dropout(dec_input)
    for dec_layer in self.layer_stack:
      dec_output = dec_layer(dec_output, x_states,
                             y_attn_mask=y_attn_mask, x_attn_mask=x_attn_mask)
    return dec_output

class Transformer(nn.Module):
  def __init__(self, hparams, init_type="uniform", *args, **kwargs):
    super(Transformer, self).__init__()

    self.hparams = hparams
    self.encoder = Encoder(hparams)
    self.dropout = nn.Dropout(hparams.dropout)
    self.decoder = Decoder(hparams)
    self.w_logit = nn.Linear(hparams.d_model, hparams.target_vocab_size, bias=False)
    if hparams.share_emb_and_softmax:
      self.w_logit.weight = self.decoder.word_emb.weight

    if self.hparams.cuda:
      self.w_logit = self.w_logit.cuda()

    init_param(self.w_logit.weight, init_type=init_type,
               init_range=self.hparams.init_range)

    if hparams.label_smoothing is not None:
      self.softmax = nn.Softmax(dim=-1)
      smooth = np.full(
        [1, 1, hparams.target_vocab_size], 1 / hparams.target_vocab_size,
        dtype=np.float32)
      self.smooth = torch.FloatTensor(smooth)
      if self.hparams.cuda:
        self.smooth = self.smooth.cuda()

  def forward(self, x_train, x_mask, x_pos_emb_indices,
              y_train, y_mask, y_pos_emb_indices, label_smoothing=True):

    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
    dec_output = self.decoder(
      enc_output, x_mask, y_train, y_mask, y_pos_emb_indices)
    dec_output = self.dropout(dec_output)
    logits = self.w_logit(dec_output)
    if label_smoothing and (self.hparams.label_smoothing is not None):
      smooth = self.hparams.label_smoothing
      probs = ((1.0 - smooth) * self.softmax(logits) +
               smooth / self.hparams.target_vocab_size)
      logits = torch.log(probs)

    return logits

  def trainable_parameters(self):
    params = self.parameters()
    return params
  
  def translate(self, x_train_batch, x_mask_batch, x_pos_emb_indices_batch,
                beam_size, max_len):

    class Hyp(object):
      def __init__(self, state=None, y=None, ctx_tm1=None, score=None):
        self.state = state
        self.y = y 
        self.ctx_tm1 = ctx_tm1
        self.score = score

    batch_size = x_train_batch.size(0)
    all_hyp, all_scores = [], []
    for i in range(batch_size): 
      x_train = x_train_batch[i, :].unsqueeze(0)
      x_mask = x_mask_batch[i, :].unsqueeze(0)
      x_pos_emb_indices = x_pos_emb_indices_batch[i, :].unsqueeze(0)

      # translate one sentence
      enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
      len_dec_seq = 0
      completed_hyp = []
      completed_hyp_scores = []
      active_hyp = [Hyp(y=[self.hparams.bos_id], score=0.)]
      length = 0
      while len(completed_hyp) < beam_size and length < max_len:
        length += 1
        new_hyp_score_list = []
        for i, hyp in enumerate(active_hyp):
          y_partial = Variable(
            torch.LongTensor(hyp.y).unsqueeze(0), volatile=True)
          y_mask = torch.ByteTensor([0] * length).unsqueeze(0)
          y_partial_pos = Variable(
            torch.arange(1, length+1).unsqueeze(0), volatile=True)
          if self.hparams.cuda:
            y_partial = y_partial.cuda()
            y_partial_pos = y_partial_pos.cuda()
            y_mask = y_mask.cuda()
          dec_output = self.decoder(
            enc_output, x_mask, y_partial, y_mask, y_partial_pos)
          dec_output = dec_output[:, -1, :]
          logits = self.w_logit(dec_output)
          probs = torch.nn.functional.softmax(logits, dim=1)
          new_hyp_scores = hyp.score + probs.data
          new_hyp_score_list.append(new_hyp_scores)
        live_hyp_num = beam_size - len(completed_hyp)
        new_hyp_scores = np.concatenate(new_hyp_score_list).flatten()
        new_hyp_pos = (-new_hyp_scores).argsort()[:live_hyp_num]
        prev_hyp_ids = new_hyp_pos / self.hparams.target_vocab_size
        word_ids = new_hyp_pos % self.hparams.target_vocab_size
        new_hyp_scores = new_hyp_scores[new_hyp_pos]

        new_hypothesis = []
        for prev_hyp_id, word_id, hyp_score in zip(
            prev_hyp_ids, word_ids, new_hyp_scores):
          prev_hyp = active_hyp[int(prev_hyp_id)]
          hyp = Hyp(y=prev_hyp.y+[int(word_id)], score=hyp_score)
          if word_id == self.hparams.eos_id:
            completed_hyp.append(hyp)
            completed_hyp_scores.append(hyp_score)
          else:
            new_hypothesis.append(hyp)
        active_hyp = new_hypothesis
      if len(completed_hyp) == 0:
        completed_hyp.append(active_hyp[0])
        completed_hyp_scores = [0.0]
      ranked_hypothesis = sorted(
        zip(completed_hyp, completed_hyp_scores),
        key=lambda x:x[1],
        reverse=True)
      h = [hyp.y for hyp, score in ranked_hypothesis]
      s = [score for hyp, score in ranked_hypothesis]
      all_hyp.append(h)
      all_scores.append(s)
    return all_hyp, all_scores     

  def translate_v1(self, x_train_batch, x_mask_batch, x_pos_emb_indices_batch,
                   beam_size, max_len):
    batch_size = x_train_batch.size(0)
    all_hyp, all_scores = [], []
    for i in range(batch_size):
      x_train = x_train_batch[i, :].unsqueeze(0)
      x_mask = x_mask_batch[i, :].unsqueeze(0)
      x_pos_emb_indices = x_pos_emb_indices_batch[i, :].unsqueeze(0)

      # translate one sentence
      enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
      len_dec_seq = 0
      hypothesis = [[self.hparams.bos_id]]
      completed_hypothesis = []
      completed_hypothesis_scores = []
      hyp_scores = Variable(torch.zeros(1), volatile=True)
      if self.hparams.cuda:
        hyp_scores = hyp_scores.cuda()
      while len(completed_hypothesis) < beam_size and len_dec_seq < max_len:
        len_dec_seq += 1
        hyp_num = len(hypothesis)
        exp_enc_output = enc_output.expand(
          hyp_num, enc_output.size(1), enc_output.size(2))
        exp_x_mask = x_mask.expand(hyp_num, x_mask.size(1))
        # (n_remain_sents * beam, seq_len)

        y_partial = torch.LongTensor(hypothesis).view(-1, len_dec_seq)
        y_partial = Variable(y_partial, volatile=True)

        y_mask = torch.ByteTensor([([0] * len_dec_seq) for _ in range(hyp_num)])
        y_partial_pos = torch.arange(1, len_dec_seq+1).unsqueeze(0)

        # size: (n_remain_sents * beam, seq_len)
        y_partial_pos = y_partial_pos.repeat(hyp_num, 1)
        y_partial_pos = Variable(
          torch.FloatTensor(y_partial_pos), volatile=True)
        if self.hparams.cuda:
          y_partial = y_partial.cuda()
          y_partial_pos = y_partial_pos.cuda()
          y_mask = y_mask.cuda()

          exp_enc_output = exp_enc_output.cuda()
          exp_x_mask = exp_x_mask.cuda()
        dec_output = self.decoder(
          exp_enc_output, exp_x_mask, y_partial, y_mask, y_partial_pos)

        # select the dec output for next word
        dec_output = dec_output[:, -1, :]
        logits = self.w_logit(dec_output)
        probs = torch.nn.functional.softmax(logits, dim=1)

        live_hyp_num = beam_size - len(completed_hypothesis)
        new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(probs) +
                          probs).view(-1)
        top_new_hyp_scores, top_new_hyp_pos = torch.topk(
          new_hyp_scores, k=live_hyp_num)
        prev_hyp_ids = top_new_hyp_pos / self.hparams.target_vocab_size
        word_ids = top_new_hyp_pos % self.hparams.target_vocab_size

        new_hypothesis = []
        live_hyp_ids = []
        new_hyp_scores = []
        for prev_hyp_id, word_id, new_hyp_score in zip(
            prev_hyp_ids.cpu().data,
            word_ids.cpu().data,
            top_new_hyp_scores.cpu().data):
          hyp_trg_words = hypothesis[prev_hyp_id] + [word_id]
          if word_id == self.hparams.eos_id:
            completed_hypothesis.append(hyp_trg_words)
            completed_hypothesis_scores.append(new_hyp_scores)
          else:
            new_hypothesis.append(hyp_trg_words)
            live_hyp_ids.append(prev_hyp_id)
            new_hyp_scores.append(new_hyp_score)
        if len(completed_hypothesis) == beam_size: break
        live_hyp_ids = torch.LongTensor(live_hyp_ids)
        if self.hparams.cuda:
          live_hyp_ids = live_hyp_ids.cuda()
        hyp_scores = Variable(torch.FloatTensor(new_hyp_scores), volatile=True)
        if self.hparams.cuda:
          hyp_scores = hyp_scores.cuda()
        hypothesis = new_hypothesis

      if len(completed_hypothesis) == 0:
        completed_hypothesis = [hypothesis[0]]
        completed_hypothesis_scores = [0.0]
      ranked_hypothesis = sorted(
        zip(completed_hypothesis, completed_hypothesis_scores),
        key=lambda x:x[1],
        reverse=True)
      h = [hyp for hyp, score in ranked_hypothesis]
      s = [score for hyp, score in ranked_hypothesis]
      all_hyp.append(h)
      all_scores.append(s)
    return all_hyp, all_scores

  # TODO(hyhieu): source corruption
  def translate_batch(self, x_train, x_mask, x_pos_emb_indices,
                      beam_size, max_len):
    """Translates a batch of sentences.

    Return:
      all_hyp: [batch_size, n_best] of hypothesis.
      all_scores: [batch_size, n_best].
    """

    # [batch_size, src_seq_len, d_model]
    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
    # print(enc_output.size())
    # sys.exit(0)

    # [batch_size * beam, src_seq_len, d_model]
    enc_output = Variable(enc_output.data.repeat(1, beam_size, 1).view(
      enc_output.size(0)*beam_size, enc_output.size(1), enc_output.size(2)))
    x_mask = x_mask.repeat(1, beam_size).view(x_mask.size(0)*beam_size,
                                              x_mask.size(1))

    batch_size, src_seq_len = x_train.size()
    trg_vocab_size = self.hparams.target_vocab_size

    beams = [Beam(beam_size, self.hparams) for _ in range(batch_size)]
    beam_to_inst = {beam_idx: inst_idx
                    for inst_idx, beam_idx in enumerate(range(batch_size))}
    n_remain_sents = batch_size
    for i in range(max_len):
      len_dec_seq = i+1 
      # (n_remain_sents * beam, seq_len)
      y_partial = torch.stack(
        [b.get_partial_y() for b in beams if not b.done]).view(-1, len_dec_seq)

      y_partial = Variable(y_partial, volatile=True)
      y_mask = torch.ByteTensor(
        [([0] * len_dec_seq) for _ in range(n_remain_sents * beam_size)])

      y_partial_pos = torch.arange(1, len_dec_seq+1).unsqueeze(0)
      # size: (n_remain_sents * beam, seq_len)
      y_partial_pos = y_partial_pos.repeat(n_remain_sents * beam_size, 1)
      y_partial_pos = Variable(torch.FloatTensor(y_partial_pos), volatile=True)

      if self.hparams.cuda:
        y_partial = y_partial.cuda()
        y_partial_pos = y_partial_pos.cuda()
        y_mask = y_mask.cuda()

        enc_output = enc_output.cuda()
        x_mask = x_mask.cuda()

      dec_output = self.decoder(
        enc_output, x_mask, y_partial, y_mask, y_partial_pos)

      # select the dec output for next word
      dec_output = dec_output[:, -1, :]
      logits = self.w_logit(dec_output)
      log_probs = torch.nn.functional.log_softmax(logits, dim=1)
      log_probs = log_probs.view(n_remain_sents, beam_size, trg_vocab_size)

      active_beam_idx_list = []
      for beam_idx in range(batch_size):
        if beams[beam_idx].done:
          continue
        inst_idx = beam_to_inst[beam_idx]
        if not beams[beam_idx].advance(log_probs.data[inst_idx], step=i):
          active_beam_idx_list += [beam_idx]

      if not active_beam_idx_list: 
        break

      active_inst_idx_list = torch.LongTensor([beam_to_inst[k] for k in active_beam_idx_list])
      if self.hparams.cuda:
        active_inst_idx_list = active_inst_idx_list.cuda()

      beam_to_inst = {beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

      enc_output = select_active_enc_info(enc_output, active_inst_idx_list,
                                          n_remain_sents, src_seq_len, self.hparams.d_model)
      x_mask = select_active_enc_mask(x_mask, active_inst_idx_list, n_remain_sents, src_seq_len)

      n_remain_sents = len(active_inst_idx_list)

    all_hyp, all_scores = [], []
    n_best = 1
    for beam in beams:
      scores, idxs = torch.sort(beam.scores, dim=0, descending=True)
      all_scores += [scores[:n_best]]
      hyps = [beam.get_y(i) for i in idxs[:n_best]]

      #hyps, scores = beam.get_hyp(n_best)
      all_hyp.append(hyps)
      all_scores.append(scores)
    return all_hyp, all_scores
    
  def debug_translate_batch(self,
                            x_train, x_mask, x_pos_emb_indices, beam_size,
                            max_len, y_train, y_mask, y_pos_emb_indices):
    # get loss
    crit = get_criterion(self.hparams)
    logits = self.forward(
      x_train, x_mask, x_pos_emb_indices,
      y_train[:, 0:1], y_mask[:, 0:1], y_pos_emb_indices[:, 0:1].contiguous(),
      label_smoothing=False)
    logits = logits.view(-1, self.hparams.target_vocab_size)
    labels = y_train[:, 1:2].contiguous().view(-1)
    #tr_loss, _ = get_performance(crit, logits[1:], labels[1:])
    # print(logits[0][labels[0]].data, labels[0].data)
    #neg_log = -torch.nn.functional.log_softmax(logits, dim=1)
    #for i in range(logits.size()[1]):
    #  print(i, neg_log[0][i].data[0])
    tr_loss, _ = get_performance(crit, logits, labels, self.hparams)
    print("train loss of first word: {}".format(tr_loss.data[0]))
    
    logits = self.forward(
      x_train, x_mask, x_pos_emb_indices,
      y_train[:, 0:2], y_mask[:, 0:2], y_pos_emb_indices[:, 0:2].contiguous(),
      label_smoothing=False)
    logits = logits.view(-1, self.hparams.target_vocab_size)
    labels = y_train[:, 1:3].contiguous().view(-1)
    #tr_loss, _ = get_performance(crit, logits[1:], labels[1:])
    print(logits[0][labels[0]].data, labels[0].data)
    tr_loss, _ = get_performance(crit, logits, labels, self.hparams)
    print("train loss of first two words: {}".format(tr_loss.data[0]))

    logits = self.forward(
      x_train, x_mask, x_pos_emb_indices,
      y_train[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous(),
      label_smoothing=False)
    logits = logits.view(-1, self.hparams.target_vocab_size)
    labels = y_train[:, 1:].contiguous().view(-1)
    tr_loss, _ = get_performance(crit, logits, labels, self.hparams)
    print("train loss of total word: {}".format(tr_loss.data[0]))
    # (batch_size, src_seq_len, d_model)

    enc_output = self.encoder(x_train, x_mask, x_pos_emb_indices)
    batch_size, src_seq_len = x_train.size()
    batch_size, trg_seq_len = y_train.size()
    trg_vocab_size = self.hparams.target_vocab_size

    dec_loss = 0.
    for i in range(trg_seq_len-1):
      len_dec = i+1
      y_partial = y_train[:, :len_dec]
      y_partial_mask = y_mask[:, :len_dec]
      y_partial_pos = y_pos_emb_indices[:, :len_dec].contiguous()
      if self.hparams.cuda:
        y_partial = y_partial.cuda()
        y_partial_pos = y_partial_pos.cuda()
        y_partial_mask = y_partial_mask.cuda()

        enc_output = enc_output.cuda()
        x_mask = x_mask.cuda()

      dec_output = self.decoder(enc_output, x_mask, y_partial, y_partial_mask, 
                                y_partial_pos)
      # select the dec output for next word
      dec_output = dec_output[:, -1, :]
      logits = self.w_logit(dec_output)
      logits = logits.view(-1, self.hparams.target_vocab_size)
      labels = y_train[:, len_dec].contiguous().view(-1)
      #if i == 0:
        #print('use pad id')
        #labels = Variable(torch.LongTensor([self.hparams.pad_id]))
        #if self.hparams.cuda: labels = labels.cuda()
        #s, indices = torch.sort(logits, descending=True)
        #print('best word', indices[0])
      loss, _ = get_performance(crit, logits, labels, self.hparams)
      print("cur_loss", loss.data)
      print("cur_labels", labels.data)
      dec_loss = dec_loss + loss.data[0]
    print('dec_loss_search: ', dec_loss)

def select_active_enc_info(enc_output, active_inst_idx_list, n_remain_sents, src_seq_len, d_model):

  original_enc_info_data = enc_output.data.view(n_remain_sents, -1, d_model).contiguous()
  active_enc_info = original_enc_info_data.index_select(0, active_inst_idx_list).contiguous()
  active_enc_info = active_enc_info.view(-1, src_seq_len, d_model).contiguous()
  return Variable(active_enc_info, volatile=True)

def select_active_enc_mask(enc_mask, active_inst_idx_list, n_remain_sents, src_seq_len):
  original_enc_mask = enc_mask.view(n_remain_sents, -1).contiguous()
  active_enc_mask = original_enc_mask.index_select(0, active_inst_idx_list).contiguous()
  active_enc_mask = active_enc_mask.view(-1, src_seq_len).contiguous()
  return active_enc_mask

class PolyNorm(object):
  def __init__(self, m=1., norm_search=True):
    self.m = m 
    self.norm_search = norm_search

  def norm_partial(self, scores_to_add, scores_so_far, next_ys):
    """
    scores_to_add: [beam_size, trg_vocab_size]
    scores_so_far: [beam_size,]
    next_ys: [beam_size, prev_trg_len]
    """
    beam_size, trg_vocab_size = scores_to_add.size()
    prev_trg_len = len(next_ys)

    scores_so_far = scores_so_far.unsqueeze(1).expand_as(scores_to_add)
    if self.norm_partial:
      return (scores_so_far * pow(prev_trg_len, self.m) + scores_to_add) / pow(prev_trg_len+1, self.m)
    else:
      return scores_so_far + scores_to_add

  def norm_complete(self, finished_list):
    if not self.norm_partial:
      for f in finished_list:
        f[1] = f[1] / pow(len(f[0]), self.m)

class Beam(object):

  def __init__(self, size, hparams):
    self.size = size 
    self.done = False
    self.hparams = hparams

    self.tt = torch.cuda if hparams.cuda else torch 

    self.scores = self.tt.FloatTensor(size).zero_()
    self.all_scores = []

    self.prev_ks = []

    self.next_ys = [self.tt.LongTensor(size).fill_(hparams.bos_id)]
    self.next_ys[0][0] = hparams.bos_id 

    self.active_hyp_size = size 
    self.active_beam_idx = [i for i in range(size)]
    self.finished = []

    self.len_norm = PolyNorm()
    #self.len_norm = None

  def advance(self, word_scores, step=0):
    """Add word to the beam.

    word_scores: (beam_size, trg_vocab_size)
    """
    trg_vocab_size = word_scores.size(1)

    if len(self.prev_ks) > 0:
      if not self.len_norm is None:
        beam_score = self.len_norm.norm_partial(word_scores, self.scores, self.next_ys)
      else:
        beam_score = self.scores.unsqueeze(1).expand_as(word_scores) + word_scores
      #for i in range(self.next_ys[-1].size(0)):
      #  if self.next_ys[-1][i] == self.hparams.eos_id:
      #    beam_score[i] = -float("inf")
    else:
      # for the first step, all rows in word_scores are the same
      beam_score = word_scores[0]
    flat_beam_score = beam_score.contiguous().view(-1)
    best_scores, best_scores_id = flat_beam_score.topk(self.size, dim=0,
                                                       largest=True,
                                                       sorted=True)
    self.all_scores.append(self.scores)
    self.scores = best_scores

    prev_k = best_scores_id / trg_vocab_size
    next_y = best_scores_id % trg_vocab_size
    self.prev_ks.append(prev_k)
    self.next_ys.append(next_y)

    if self.next_ys[-1][0] == self.hparams.eos_id:
      # self.active_hyp_size -= 1
      # self.finished.append((self.get_y(0), self.scores[0]))
      # self.done = (self.active_hyp_size == 0)

      self.done = True
    return self.done

  def get_partial_y(self):
    """Return the partial y hypothesis for this beam."""

    if len(self.next_ys) == 1:
      dec_seq = self.next_ys[0].unsqueeze(1)
    else:
      _, keys = torch.sort(self.scores, dim=0, descending=True)
      hyps = [self.get_y(k) for k in keys]
      hyps = [[self.hparams.bos_id] + h for h in hyps]
      dec_seq = torch.from_numpy(np.array(hyps))
    return dec_seq

  def get_y(self, k):
    """k: the index in the beam to construct."""

    hyp = []
    for j in range(len(self.prev_ks)-1, -1, -1):
      hyp.append(self.next_ys[j+1][k])
      k = self.prev_ks[j][k]
    return hyp[::-1]

  def get_hyp(self, n=1):
    if len(self.finished) < self.size:
      _, keys = torch.sort(self.scores, dim=0, descending=True)
      for k in keys[:(self.size-len(self.finished))]:
        self.finished.append((self.get_y(k), self.scores[k]))
    if not self.len_norm is None:
      self.norm_complete(self.finished)
    self.finished.sort(key=lambda a: -a[1])
    scores = [s for _, s in self.finished]
    hyps = [h for h, _ in self.finished]
    return hyps, scores
    
