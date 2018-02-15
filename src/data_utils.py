from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import shutil
import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from hparams import Iwslt16EnDeBpe16Params
from hparams import Iwslt16EnDeBpe32Params
from hparams import Iwslt16EnDeTinyParams

class DataLoader(object):
  def __init__(self, hparams="tiny"):
    """Encloses both train and valid data.

    Args:
      hparams: must be ['tiny' 'bpe16' 'bpe32']
    """

    if hparams == "tiny":
      self.hparams = Iwslt16EnDeTinyParams()
    elif hparams == "bpe16":
      self.hparams = Iwslt16EnDeBpe16Params()
    elif hparams == "bpe32":
      self.hparams = Iwslt16EnDeBpe16Params()
    else:
      raise ValueError("Unknown hparams set '{0}'".format(hparams))

    print("-" * 80)
    print("Building data for '{0}' from '{1}'".format(
      self.hparams.dataset, self.hparams.data_path))

    # vocab
    (self.source_word_to_index,
     self.source_index_to_word) = self._build_vocab(self.hparams.source_vocab)

    (self.target_word_to_index,
     self.target_index_to_word) = self._build_vocab(self.hparams.target_vocab)

    # train data
    self.x_train, self.y_train = self._build_parallel(self.hparams.source_train,
                                                      self.hparams.target_train,
                                                      is_training=True)
    self._shuffle()
    self.train_index = 0
    self.train_size = len(self.x_train)

    # valid data
    self.x_valid, self.y_valid = self._build_parallel(self.hparams.source_valid,
                                                      self.hparams.target_valid,
                                                      is_training=False)
    self.valid_index = 0
    self.valid_size = len(self.x_valid)

  def next_valid(self, valid_batch_size=20):
    """Retrieves a sentence of testing examples.

    Returns:
      (x_valid, x_len): a pair of torch Tensors of size [batch, source_length]
        and [batch_size].
      (y_valid, y_len): a pair of torch Tensors of size [batch, target_length]
        and [batch_size].
      end_of_epoch: whether we reach the end of training examples.
    """

    end_of_epoch = False
    start_index = self.valid_index
    end_index = min(start_index + valid_batch_size, self.valid_size)
    batch_size = end_index - start_index

    # pad data
    x_valid = self.x_valid[start_index : end_index]
    y_valid = self.y_valid[start_index : end_index]
    x_valid, x_mask, x_pos_emb_indices, x_count = self._pad(sentences=x_valid)
    y_valid, y_mask, y_pos_emb_indices, y_count = self._pad(sentences=y_valid)

    # shuffle if reaches the end of data
    if end_index >= self.valid_size:
      end_of_epoch = True
      self._shuffle()
      self.valid_index = 0
    else:
      self.valid_index += batch_size

    return ((x_valid, x_mask, x_pos_emb_indices, x_count),
            (y_valid, y_mask, y_pos_emb_indices, y_count),
            end_of_epoch)

  def next_train(self):
    """Retrieves a batch of training examples.

    Returns:
      (x_train, x_len): a pair of torch Tensors of size [batch, source_length]
        and [batch_size].
      (y_train, y_len): a pair of torch Tensors of size [batch, target_length]
        and [batch_size].
      end_of_epoch: whether we reach the end of training examples.
    """

    end_of_epoch = False
    start_index = self.train_index
    end_index = min(start_index + self.hparams.batch_size, self.train_size)
    batch_size = end_index - start_index

    # pad data
    x_train = self.x_train[start_index : end_index]
    y_train = self.y_train[start_index : end_index]
    x_train, x_mask, x_pos_emb_indices, x_count = self._pad(sentences=x_train)
    y_train, y_mask, y_pos_emb_indices, y_count = self._pad(sentences=y_train)

    # shuffle if reaches the end of data
    if end_index >= self.train_size:
      end_of_epoch = True
      self._shuffle()
      self.train_index = 0
    else:
      self.train_index += batch_size

    return ((x_train, x_mask, x_pos_emb_indices, x_count),
            (y_train, y_mask, y_pos_emb_indices, y_count),
            end_of_epoch)

  def _pad(self, sentences, volatile=False):
    """Pad all instances in [data] to the longest length.

    Args:
      sentences: list of [batch_size] lists.

    Returns:
      padded_sentences: Variable of size [batch_size, max_len], the sentences.
      mask: Variable of size [batch_size, max_len]. 1 means to ignore.
      pos_emb_indices: Variable of size [batch_size, max_len]. indices to use
        when computing positional embedding.
      sum_len: total words
    """

    lengths = [len(sentence) for sentence in sentences]
    sum_len = sum(lengths)
    max_len = max(lengths)

    padded_sentences = [
      sentence + ([self.hparams.pad_id] * (max_len - len(sentence)))
      for sentence in sentences]
    mask = [
      ([0] * len(sentence)) + ([1] * (max_len - len(sentence)))
      for sentence in sentences]
    pos_emb_indices = [
      [i for i in range(len(sentence))] + ([-1] * (max_len - len(sentence)))
      for sentence in sentences
    ]

    padded_sentences = Variable(torch.LongTensor(padded_sentences),
                                volatile=volatile)
    mask = torch.ByteTensor(mask)
    pos_emb_indices = Variable(torch.FloatTensor(pos_emb_indices),
                               volatile=volatile)

    if self.hparams.cuda:
      padded_sentences = padded_sentences.cuda()
      mask = mask.cuda()
      pos_emb_indices = pos_emb_indices.cuda()

    return padded_sentences, mask, pos_emb_indices, sum_len

  def _shuffle(self, verbose=False):
    """Shuffle (x_train, y_train)."""

    if verbose:
      print("Shuffle training data")
    xy_train = list(zip(self.x_train, self.y_train))
    random.shuffle(xy_train)
    self.x_train, self.y_train = zip(*xy_train)

  def _build_parallel(self, source_file, target_file, is_training):
    """Build pair of data."""

    print("-" * 80)
    print("Loading parallel data from '{0}' and '{1}'".format(
      source_file, target_file))

    source_file = os.path.join(self.hparams.data_path, source_file)
    with open(source_file) as finp:
      source_lines = finp.read().split("\n")

    target_file = os.path.join(self.hparams.data_path, target_file)
    with open(target_file) as finp:
      target_lines = finp.read().split("\n")

    source_data, target_data = [], []
    total_sents = 0
    for i, (source_line, target_line) in enumerate(
        zip(source_lines, target_lines)):
      source_line = source_line.strip()
      target_line = target_line.strip()
      if not source_line or not target_line:
        continue

      source_indices, target_indices = [], []
      source_tokens = source_line.split(" ")
      target_tokens = target_line.split(" ")
      if (is_training and
          (len(source_line) > self.hparams.max_train_len or
           len(target_line) > self.hparams.max_train_len)):
        continue

      total_sents += 1

      for source_token in source_tokens:
        source_token = source_token.strip()
        if source_token not in self.source_word_to_index:
          source_token = self.hparams.unk
        source_index = self.source_word_to_index[source_token]
        source_indices.append(source_index)

      for target_token in target_tokens:
        target_token = target_token.strip()
        if target_token not in self.target_word_to_index:
          target_token = self.hparams.unk
        target_index = self.target_word_to_index[target_token]
        target_indices.append(target_index)

      assert source_indices[-1] == self.hparams.eos_id
      assert target_indices[-1] == self.hparams.eos_id

      source_data.append(source_indices)
      target_data.append(target_indices)

      if (self.hparams.train_limit is not None and
          self.hparams.train_limit <= total_sents):
        break

      if total_sents % 10000 == 0:
        print("{0:>6d} pairs".format(total_sents))

    assert len(source_data) == len(target_data)
    print("{0:>6d} pairs".format(len(source_data)))

    return source_data, target_data

  def _build_vocab(self, file_name):
    """Build word_to_index and index_to word dicts."""

    print("-" * 80)
    print("Loading vocab from '{0}'".format(file_name))
    file_name = os.path.join(self.hparams.data_path, file_name)
    with open(file_name) as finp:
      lines = finp.read().split("\n")

    word_to_index, index_to_word = {}, {}
    for line in lines:
      line = line.strip()
      if not line:
        continue
      word, index = line.split("\t")
      if not word or not index:
        continue
      if index.startswith("-"):
        index = -int(index)
      elif word == self.hparams.unk:
        index = self.hparams.unk_id
      elif word == self.hparams.eos:
        index = self.hparams.eos_id
      elif word == self.hparams.bos:
        index = self.hparams.bos_id
      else:
        raise ValueError("Wrong word '{0}' or index '{1}'".format(word, index))
      word_to_index[word] = index
      index_to_word[index] = word

    assert len(word_to_index) == len(index_to_word)
    print("Done. vocab_size = {0}".format(len(word_to_index)))

    return word_to_index, index_to_word

