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
  weight = torch.ones(hparams.vocab_size)
  weight[hparams.pad_id] = 0
  crit = nn.CrossEntropyLoss(weight, size_average=False,
                             ignore_index=hparams.pad_id)
  if hparams.cuda:
    crit = crit.cuda()
  return crit


def get_performance(crit, logits, labels):
  loss = crit(logits, labels)
  _, preds = torch.max(logits, dim=1)
  acc = torch.eq(preds, labels).sum()
  return loss, acc


def get_attn_padding_mask(seq_q, seq_k, pad_id=0):
  """Indicate the padding-related part to mask.

  Args:
    seq_q: Torch tensor [batch_size, len_q].
    seq_k: Torch tensor [batch_size, len_k].

  Returns:
    attn_padding_mask: ByteTensor [batch_size, len_q, len_k]. 1 means to the
      corresponding position is pad.
  """

  assert seq_q.dim() == 2 and seq_k.dim() == 2
  batch_size_q, len_q = seq_q.size()
  batch_size_k, len_k = seq_k.size()

  assert batch_size_q == batch_size_k

  # [batch_size, 1, len_k] -> [batch_size, len_q, len_k]
  attn_padding_mask = seq_k.data.eq(pad_id).unsqueeze(1).expand(-1, len_q, -1)

  return attn_padding_mask

def get_attn_subsequent_mask(seq, pad_id=0):
  """ Get an attention mask to avoid using the subsequent info."""

  assert seq.dim() == 2
  attn_shape = (seq.size(0), seq.size(1), seq.size(1))
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype(np.uint8)
  subsequent_mask = torch.from_numpy(subsequent_mask)
  if seq.is_cuda:
    subsequent_mask = subsequent_mask.cuda()
  return subsequent_mask


def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr


def count_params(params):
  num_params = 0
  for param in params:
    num_params += param.numel()
  return num_params

