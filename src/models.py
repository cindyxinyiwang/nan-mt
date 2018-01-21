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


class Encoder(nn.Module):
  def __init__(self, *args, **kwargs):
    raise NotImplementedError("Bite me!")

  def forward(self, source_indices, source_lengths, *args, **kwargs):
    """Performs a forward pass.

    Args:
      source_indices: Torch Tensor of size [batch_size, max_len]
    """
    raise NotImplementedError("Bite me!")


class Decoder(nn.Module):
  def __init__(self, *args, **kwargs):
    raise NotImplementedError("Bite me!")

  def forward(self, source_states):
    """Performs a forward pass.

    Args:
      source_states: Torch Tensor of size [batch_size, max_len]
    """
    raise NotImplementedError("Bite me!")

