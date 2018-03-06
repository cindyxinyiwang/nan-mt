from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


def add_argument(parser, name, type, default, help):
  """Add an argument.

  Args:
    name: arg's name.
    type: must be ["bool", "int", "float", "str"].
    default: corresponding type of value.
    help: help message.
  """

  if type == "bool":
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument("--{0}".format(name), dest=name,
                                action="store_true", help=help)
    feature_parser.add_argument("--no_{0}".format(name), dest=name,
                                action="store_false", help=help)
    parser.set_defaults(name=default)
  elif type == "int":
    parser.add_argument("--{0}".format(name),
                        type=int, default=default, help=help)
  elif type == "float":
    parser.add_argument("--{0}".format(name),
                        type=float, default=default, help=help)
  elif type == "str":
    parser.add_argument("--{0}".format(name),
                        type=str, default=default, help=help)
  else:
    raise ValueError("Unknown type '{0}'".format(type))


def save_checkpoint(step, model, optimizer, hparams, path):
  print("Saving model to '{0}'".format(path))
  torch.save(step, os.path.join(path, "step.pt"))
  torch.save(model, os.path.join(path, "model.pt"))
  torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
  torch.save(hparams, os.path.join(path, "hparams.pt"))


class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()


def get_criterion(hparams):
  weights = torch.ones(hparams.trg_vocab_size)
  weights[hparams.pad_id] = 0
  #print(weights)
  #crit = nn.CrossEntropyLoss(weights, size_average=False, reduce=False)
  crit = nn.CrossEntropyLoss(size_average=False, ignore_index=hparams.pad_id)
  #crit = nn.NLLLoss(size_average=False, ignore_index=hparams.pad_id)
  if hparams.cuda:
    crit = crit.cuda()
  return crit


def get_performance(crit, logits, labels, hparams):
  mask = (labels == hparams.pad_id)
  #loss = crit(logits, labels).masked_fill_(mask, 0).sum()
  loss = crit(logits, labels)
  '''
  print(logits.data)
  ls = nn.LogSoftmax()
  ll = ls(logits)
  print(ll.data)
  for i in range(labels.size()[0]):
    print(i, ll[i][labels[i]].data, labels[i])
  loss = crit(ll, labels)
  print(loss.data)
  '''
  _, preds = torch.max(logits, dim=1)
  acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()
  return loss, acc

def get_attn_padding_mask(seq_q, seq_k, pad_id):
  ''' Indicate the padding-related part to mask '''
  assert seq_q.dim() == 2 and seq_k.dim() == 2
  mb_size, len_q = seq_q.size()
  mb_size, len_k = seq_k.size()
  pad_attn_mask = seq_k.data.eq(pad_id).unsqueeze(1)   # bx1xsk
  pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
  return pad_attn_mask

def get_attn_subsequent_mask(seq, pad_id):
  ''' Get an attention mask to avoid using the subsequent info.'''
  assert seq.dim() == 2
  attn_shape = (seq.size(0), seq.size(1), seq.size(1))
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  subsequent_mask = torch.from_numpy(subsequent_mask)
  if seq.is_cuda:
      subsequent_mask = subsequent_mask.cuda()
  return subsequent_mask

def position_encoding_init(n_position, d_pos_vec):
  ''' Init the sinusoid position encoding table '''
  # keep dim 0 for padding token position encoding zero vector
  position_enc = np.array([
      [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
      if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
  position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
  position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
  return torch.from_numpy(position_enc).type(torch.FloatTensor)

'''
def get_attn_subsequent_mask(seq, pad_id=0):
  """ Get an attention mask to avoid using the subsequent info."""

  assert seq.dim() == 2
  batch_size, max_len = seq.size()
  sub_mask = torch.triu(
    torch.ones(max_len, max_len), diagonal=1).unsqueeze(0).repeat(
      batch_size, 1, 1).type(torch.ByteTensor)
  if seq.is_cuda:
    sub_mask = sub_mask.cuda()
  return sub_mask
'''

def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr


def count_params(params):
  num_params = 0
  for param in params:
    num_params += param.numel()
  return num_params


class ScheduledOptim(object):
  '''A simple wrapper class for learning rate scheduling'''

  def __init__(self, optimizer, d_model, n_warmup_steps):
      self.optimizer = optimizer
      self.d_model = d_model
      self.n_warmup_steps = n_warmup_steps
      self.n_current_steps = 0
      self.lr = 0.001

  def step(self):
      "Step by the inner optimizer"
      self.optimizer.step()

  def zero_grad(self):
      "Zero out the gradients by the inner optimizer"
      self.optimizer.zero_grad()

  def update_learning_rate(self):
      ''' Learning rate scheduling per step '''

      self.n_current_steps += 1
      new_lr = np.power(self.d_model, -0.5) * np.min([
          np.power(self.n_current_steps, -0.5),
          np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
      self.lr = new_lr
      for param_group in self.optimizer.param_groups:
          param_group['lr'] = new_lr
