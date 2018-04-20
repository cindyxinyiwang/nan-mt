from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import random
import shutil
import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from hparams import Iwslt16EnDeBpe32SharedParams

class DataLoader(object):
  def __init__(self, hparams, decode=False):
    """Encloses both train and valid data.

    Args:
      hparams: must be ['tiny' 'bpe16' 'bpe32']
    """

    self.hparams = hparams
    self.decode = decode

    print("-" * 80)
    print("Building data for '{0}' from '{1}'".format(
      self.hparams.dataset, self.hparams.data_path))

    # vocab
    (self.source_word_to_index,
     self.source_index_to_word) = self._build_vocab(self.hparams.source_vocab)
    self.source_vocab_size = len(self.source_word_to_index)

    (self.target_word_to_index,
     self.target_index_to_word) = self._build_vocab(self.hparams.target_vocab)
    self.target_vocab_size = len(self.target_word_to_index)

    assert (self.source_word_to_index[self.hparams.pad] ==
            self.target_word_to_index[self.hparams.pad])
    assert (self.source_word_to_index[self.hparams.unk] ==
            self.target_word_to_index[self.hparams.unk])
    assert (self.source_word_to_index[self.hparams.bos] ==
            self.target_word_to_index[self.hparams.bos])
    assert (self.source_word_to_index[self.hparams.eos] ==
            self.target_word_to_index[self.hparams.eos])

    if (self.hparams.raml_source or (hasattr(self.hparams, "raml_target")
                                     and self.hparams.raml_target)):
      self.softmax = torch.nn.Softmax(dim=-1)

    if self.decode:
      self.x_test, self.y_test = self._build_parallel(
        self.hparams.source_test, self.hparams.target_test, is_training=False)
      self.test_size = len(self.x_test)
      self.reset_test()
      return
    else:
      # train data
      self.x_train, self.y_train = self._build_parallel(
        self.hparams.source_train, self.hparams.target_train, is_training=True,
        sort=True)

      # signifies that x_train, y_train are not ready for batching
      self.train_size = len(self.x_train)
      self.n_train_batches = None  
      self.reset_train()

      # valid data
      self.x_valid, self.y_valid = self._build_parallel(
        self.hparams.source_valid, self.hparams.target_valid, is_training=False)
      self.valid_size = len(self.x_valid)
      self.reset_valid()

  def reset_train(self):
    """Shuffle training data. Prepare the batching scheme if necessary."""

    if self.hparams.batcher == "word":
      if self.n_train_batches is None:
        start_indices, end_indices = [], []
        start_index = 0
        while start_index < self.train_size:
          end_index = start_index
          word_count = 0
          while (end_index + 1 < self.train_size and
                 (word_count +
                  len(self.x_train[end_index + 1]) +
                  len(self.y_train[end_index + 1])) <= self.hparams.batch_size):
            end_index += 1
            word_count += (len(self.x_train[end_index]) +
                           len(self.y_train[end_index]))
          start_indices.append(start_index)
          end_indices.append(end_index + 1)
          start_index = end_index + 1
        assert len(start_indices) == len(end_indices)
        self.n_train_batches = len(start_indices)
        self.start_indices = start_indices
        self.end_indices = end_indices
    elif self.hparams.batcher == "sent":
      if self.n_train_batches is None:
        self.n_train_batches = ((self.train_size + self.hparams.batch_size - 1)
                                // self.hparams.batch_size)
    else:
      raise ValueError("Unknown batcher scheme '{0}'".format(self.batcher))
    self.train_queue = np.random.permutation(self.n_train_batches)
    self.train_index = 0

  def reset_valid(self):
    self.valid_index = 0

  def reset_test(self):
    self.test_index = 0

  def next_test(self, test_batch_size=1, raml=False):
    end_of_epoch = False
    start_index = self.test_index
    end_index = min(start_index + test_batch_size, self.test_size)
    batch_size = end_index - start_index

    # pad data
    x_test = self.x_test[start_index: end_index]
    y_test = self.y_test[start_index: end_index]
    if raml:
      x_test = [sent for sent in x_test for _ in range(self.hparams.n_corrupts)]
      x_test_raml, x_test, x_mask, x_pos_emb_indices, x_count = self._pad(
        sentences=x_test, pad_id=self.pad_id, raml=True,
        vocab_size=self.hparams.source_vocab_size, volatile=True, pad_corrupt=self.hparams.src_pad_corrupt)
    else:
      x_test, x_mask, x_pos_emb_indices, x_count = self._pad(
        sentences=x_test, pad_id=self.pad_id, volatile=True)

    y_test, y_mask, y_pos_emb_indices, y_count = self._pad(
      sentences=y_test, pad_id=self.pad_id, volatile=True)

    if end_index >= self.test_size:
      end_of_epoch = True
      self.test_index = 0
    else:
      self.test_index += batch_size

    if raml:
      return ((x_test_raml, x_test, x_mask, x_pos_emb_indices, x_count),
              (y_test, y_mask, y_pos_emb_indices, y_count),
              batch_size, end_of_epoch)
    else:
      return ((x_test, x_mask, x_pos_emb_indices, x_count),
              (y_test, y_mask, y_pos_emb_indices, y_count),
              batch_size, end_of_epoch)

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
    x_valid, x_mask, x_pos_emb_indices, x_count = self._pad(
      sentences=x_valid, pad_id=self.pad_id, volatile=True)
    y_valid, y_mask, y_pos_emb_indices, y_count = self._pad(
      sentences=y_valid, pad_id=self.pad_id, volatile=True)

    # shuffle if reaches the end of data
    if end_index >= self.valid_size:
      end_of_epoch = True
      self.valid_index = 0
    else:
      self.valid_index += batch_size

    return ((x_valid, x_mask, x_pos_emb_indices, x_count),
            (y_valid, y_mask, y_pos_emb_indices, y_count),
            batch_size, end_of_epoch)

  def next_train(self):
    """Retrieves a batch of training examples.

    Returns:
      (x_train, x_len): a pair of torch Tensors of size [batch, source_length]
        and [batch_size].
      (y_train, y_len): a pair of torch Tensors of size [batch, target_length]
        and [batch_size].
      end_of_epoch: whether we reach the end of training examples.
    """
    if self.hparams.batcher == "word":
      start_index = self.start_indices[self.train_queue[self.train_index]]
      end_index = self.end_indices[self.train_queue[self.train_index]]
    elif self.hparams.batcher == "sent":
      start_index = (self.train_queue[self.train_index] *
                     self.hparams.batch_size)
      end_index = min(start_index + self.hparams.batch_size, self.train_size)
    else:
      raise ValueError("Unknown batcher '{0}'".format(self.hparams.batcher))

    x_train = self.x_train[start_index : end_index]
    y_train = self.y_train[start_index : end_index]
    self.train_index += 1
    batch_size = len(x_train)

    # pad data
    if self.hparams.raml_source:
      x_train_raml, x_train, x_mask, x_pos_emb_indices, x_count = self._pad(
        sentences=x_train, pad_id=self.pad_id, raml=True,
        vocab_size=self.hparams.source_vocab_size, pad_corrupt=self.hparams.src_pad_corrupt)
    else:
      x_train, x_mask, x_pos_emb_indices, x_count = self._pad(
        sentences=x_train, pad_id=self.pad_id)

    if self.hparams.raml_target:
      y_train_raml, y_train, y_mask, y_pos_emb_indices, y_count = self._pad(
        sentences=y_train, pad_id=self.pad_id, raml=True,
        vocab_size=self.hparams.target_vocab_size, pad_corrupt=self.hparams.trg_pad_corrupt)
    else:
      y_train, y_mask, y_pos_emb_indices, y_count = self._pad(
        sentences=y_train, pad_id=self.pad_id)

    # shuffle if reaches the end of data
    if self.train_index > self.n_train_batches - 1:
      self.reset_train()

    if self.hparams.raml_source and self.hparams.raml_target:
      return ((x_train_raml, x_train, x_mask, x_pos_emb_indices, x_count),
              (y_train_raml, y_train, y_mask, y_pos_emb_indices, y_count),
              batch_size)

    if self.hparams.raml_source:
      return ((x_train_raml, x_train, x_mask, x_pos_emb_indices, x_count),
              (y_train, y_mask, y_pos_emb_indices, y_count),
              batch_size)

    if self.hparams.raml_target:
      return ((x_train, x_mask, x_pos_emb_indices, x_count),
              (y_train_raml, y_train, y_mask, y_pos_emb_indices, y_count),
              batch_size)

    return ((x_train, x_mask, x_pos_emb_indices, x_count),
            (y_train, y_mask, y_pos_emb_indices, y_count),
            batch_size)

  def _pad(self, sentences, pad_id, volatile=False,
           raml=False, vocab_size=None, pad_corrupt=False):
    """Pad all instances in [data] to the longest length.

    Args:
      sentences: list of [batch_size] lists.

    Returns:
      padded_sentences: Variable of size [batch_size, max_len], the sentences.
      mask: Variable of size [batch_size, max_len]. 1 means to ignore.
      pos_emb_indices: Variable of size [batch_size, max_len]. indices to use
        when computing positional embedding.
      sum_len: total number of words.
      raml: if True, the sentences will be replaced by their samples according
        to the exp payoff distribution, ie. exp(-HammingDist(s, sents)).
      vocab_size: if raml is True, vocab_size has to be specified for negative
        sampling.
    """

    batch_size = len(sentences)
    lengths = [len(sentence) for sentence in sentences]
    sum_len = sum(lengths)
    max_len = max(lengths)

    padded_sentences = [
      sentence + ([pad_id] * (max_len - len(sentence)))
      for sentence in sentences]
    mask = [
      ([0] * len(sentence)) + ([1] * (max_len - len(sentence)))
      for sentence in sentences]
    pos_emb_indices = [
      [i+1 for i in range(len(sentence))] + ([0] * (max_len - len(sentence)))
      for sentence in sentences
    ]

    padded_sentences = Variable(torch.LongTensor(padded_sentences))
    mask = torch.ByteTensor(mask)
    pos_emb_indices = Variable(torch.FloatTensor(pos_emb_indices))

    if self.hparams.cuda:
      padded_sentences = padded_sentences.cuda()
      mask = mask.cuda()
      pos_emb_indices = pos_emb_indices.cuda()

    if not raml:
      return padded_sentences, mask, pos_emb_indices, sum_len

    assert vocab_size is not None
    # first, sample the number of words to corrupt for each sentence
    logits = torch.arange(max_len)
    if self.hparams.cuda:
      logits = logits.cuda()
    logits = logits.mul_(-1).unsqueeze(0).expand_as(
      padded_sentences).contiguous().masked_fill_(mask, -self.hparams.inf)
    logits = Variable(logits, volatile=True)
    if self.hparams.cuda:
      logits = logits.cuda()
    probs = self.softmax(logits.mul_(self.hparams.raml_tau))
    num_words = torch.distributions.Categorical(probs).sample()

    # sample the indices
    lengths = torch.FloatTensor(lengths)
    if self.hparams.cuda:
      lengths = lengths.cuda()

    corrupt_pos = num_words.data.float().div_(lengths).unsqueeze(
      1).expand_as(padded_sentences).contiguous().masked_fill_(mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte()
    total_words = int(corrupt_pos.sum())

    # sample the corrupts, which will be added to padded_sentences
    corrupt_val = torch.LongTensor(total_words)
    if pad_corrupt:
      corrupt_val.fill_(self.hparams.pad_id)
    else:
      corrupt_val = corrupt_val.random_(0, vocab_size-1)
    corrupts = torch.zeros(batch_size, max_len).long()
    if self.hparams.cuda:
      corrupt_val = corrupt_val.long().cuda()
      corrupts = corrupts.cuda()
      corrupt_pos = corrupt_pos.cuda()
    corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)

    sample_sentences = padded_sentences.add(Variable(corrupts)).remainder_(
      vocab_size).masked_fill_(Variable(mask), pad_id)

    return sample_sentences, padded_sentences, mask, pos_emb_indices, sum_len

  def _build_parallel(self, source_file, target_file, is_training, sort=False):
    """Build pair of data."""

    print("-" * 80)
    print("Loading parallel data from '{0}' and '{1}'".format(
      source_file, target_file))

    source_file = os.path.join(self.hparams.data_path, source_file)
    with open(source_file, encoding='utf-8') as finp:
      source_lines = finp.read().split("\n")

    target_file = os.path.join(self.hparams.data_path, target_file)
    with open(target_file, encoding='utf-8') as finp:
      target_lines = finp.read().split("\n")

    source_data, target_data = [], []
    source_lens = []
    total_sents = 0
    source_unk_count, target_unk_count = 0, 0
    for i, (source_line, target_line) in enumerate(
        zip(source_lines, target_lines)):
      source_line = source_line.strip()
      target_line = target_line.strip()
      if not source_line or not target_line:
        continue

      source_indices, target_indices = [self.bos_id], [self.bos_id]
      source_tokens = source_line.split(" ")
      target_tokens = target_line.split(" ")
      if is_training and len(target_line) > self.hparams.max_len:
        continue

      total_sents += 1

      for source_token in source_tokens:
        source_token = source_token.strip()
        if source_token not in self.source_word_to_index:
          source_token = self.hparams.unk
          source_unk_count += 1
        source_index = self.source_word_to_index[source_token]
        source_indices.append(source_index)

      for target_token in target_tokens:
        target_token = target_token.strip()
        if target_token not in self.target_word_to_index:
          target_token = self.hparams.unk
          target_unk_count += 1
        target_index = self.target_word_to_index[target_token]
        target_indices.append(target_index)

      source_indices.append(self.eos_id)
      target_indices.append(self.eos_id)

      source_lens.append(len(source_indices))
      source_data.append(source_indices)
      target_data.append(target_indices)

      if (self.hparams.n_train_sents is not None and
          self.hparams.n_train_sents <= total_sents):
        break

      if total_sents % 10000 == 0:
        print("{0:>6d} pairs. src_unk={1}. tgt_unk={2}".format(
          total_sents, source_unk_count, target_unk_count))

    assert len(source_data) == len(target_data)
    print("{0:>6d} pairs. src_unk={1}. tgt_unk={2}".format(
      total_sents, source_unk_count, target_unk_count))

    if sort:
      print("Heuristic sort based on source lens")
      indices = np.argsort(source_lens)
      source_data = [source_data[index] for index in indices]
      target_data = [target_data[index] for index in indices]

    return np.array(source_data), np.array(target_data)

  def _build_vocab(self, file_name):
    """Build word_to_index and index_to word dicts."""

    print("-" * 80)
    print("Loading vocab from '{0}'".format(file_name))
    file_name = os.path.join(self.hparams.data_path, file_name)
    with open(file_name, encoding='utf-8') as finp:
      lines = finp.read().split("\n")

    word_to_index, index_to_word = {}, {}
    for line in lines:
      line = line.strip()
      if not line:
        continue
      word_index = line.split("▁")
      if len(word_index) != 2:
        print("Weird line: '{0}'. split_len={1}".format(line, len(word_index)))
        continue
      word, index = word_index
      index = int(index)
      word_to_index[word] = index
      index_to_word[index] = word
      if word == self.hparams.unk:
        self.unk_id = index
      elif word == self.hparams.bos:
        self.bos_id = index
      elif word == self.hparams.eos:
        self.eos_id = index
      elif word == self.hparams.pad:
        self.pad_id = index

    assert len(word_to_index) == len(index_to_word), (
      "|word_to_index|={0} != |index_to_word|={1}".format(len(word_to_index),
                                                        len(index_to_word)))
    print("Done. vocab_size = {0}".format(len(word_to_index)))

    return word_to_index, index_to_word

