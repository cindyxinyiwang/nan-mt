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
import torch.nn.init as init


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


def save_checkpoint(extra, model, optimizer, hparams, path, save_model_name="model"):
  print("Saving model to '{0}'".format(path))
  torch.save(extra, os.path.join(path, "extra.pt"))
  torch.save(model, os.path.join(path, save_model_name+".pt"))
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
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, size_average=False)
  if hparams.cuda:
    crit = crit.cuda()
  return crit


def get_performance(crit, logits, labels, hparams):
  mask = (labels == hparams.pad_id)
  loss = crit(logits, labels)
  _, preds = torch.max(logits, dim=1)
  acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()

  return loss, acc


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


def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr


def count_params(params):
  num_params = sum(p.data.nelement() for p in params)
  return num_params


def init_param(p, init_type="xavier_normal", init_range=None):
  if init_type == "xavier_normal":
    init.xavier_normal(p)
  elif init_type == "xavier_uniform":
    init.xavier_uniform(p)
  elif init_type == "kaiming_normal":
    init.kaiming_normal(p)
  elif init_type == "kaiming_uniform":
    init.kaiming_uniform(p)
  elif init_type == "uniform":
    assert init_range is not None and init_range > 0
    init.uniform(p, -init_range, init_range)
  else:
    raise ValueError("Unknown init_type '{0}'".format(init_type))


def grad_clip(params, grad_bound=None):
  """Clipping gradients at L-2 norm grad_bound. Returns the L-2 norm."""

  params = list(filter(lambda p: p.grad is not None, params))
  total_norm = 0
  for p in params:
    if p.grad is None:
      continue
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm ** 2
  total_norm = total_norm ** 0.5

  if grad_bound is not None:
    clip_coef = grad_bound / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in params:
        p.grad.data.mul_(clip_coef)
  return total_norm

