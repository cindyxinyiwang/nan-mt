from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import _pickle as pickle
import shutil
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataLoader

from hparams import Iwslt16EnDeBpe16Params
from hparams import Iwslt16EnDeTinyParams

from utils import *
from models import *

parser = argparse.ArgumentParser(description="Neural MT")

parser.add_argument("--reset_output_dir", type=bool, default="reset_output_dir",
                    help="delete output directory if it exists")
parser.add_argument("--output_dir", type=str, default="outputs",
                    help="path to output directory")
parser.add_argument("--log_every", type=int, default="50",
                    help="name of pre-processed file")


args = parser.parse_args()


def train():
  hparams = Iwslt16EnDeTinyParams()
  data = DataLoader()

  # build model
  model = Transformer(hparams=hparams)
  # TODO(hyhieu): verify how are log_probs summed and averaged
  crit = get_criterion(hparams)
  optim = torch.optim.Adam(model.trainable_parameters(),
                           lr=hparams.learning_rate,
                           betas=(0.9, 0.98), eps=1e-09)

  # train loop
  print("-" * 80)
  print("Start training")
  for epoch in range(hparams.num_epochs):
    start_time = time.time()
    step = 0
    target_words = 0

    # training activities
    while True:
      # next batch
      ((x_train, x_mask, x_pos_emb_indices, x_count),
       (y_train, y_mask, y_pos_emb_indices, y_count),
       end_of_epoch) = data.next_train()

      # word count
      target_words += y_count

      # forward pass
      optim.zero_grad()
      logits = model.forward(
        x_train, x_mask, x_pos_emb_indices,
        y_train[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous())
      logits = logits.view(-1, hparams.vocab_size)
      labels = y_train[:, 1:].contiguous().view(-1)
      train_loss = crit(logits, labels).div(hparams.batch_size)

      # backward pass and training
      train_loss.backward()
      optim.step()

      step += 1
      if step % args.log_every == 0:
        curr_time = time.time()
        elapsed = (curr_time - start_time) / 60.0
        log_string = "step={0:<5d}".format(step)
        log_string += " mins={0:<5.2f}".format(elapsed)
        log_string += " tr_loss={0:<7.2f}".format(train_loss.data[0])
        log_string += " wpm={0:<5.2f}".format(target_words / elapsed)
        print(log_string)

      if end_of_epoch:
        break

    # End-of-Epoch activites, e.g: compute PPL, BLEU, etc.
    valid_words = 0
    valid_loss = 0
    while True:
      # next batch
      ((x_valid, x_mask, x_pos_emb_indices, x_count),
       (y_valid, y_mask, y_pos_emb_indices, y_count),
       end_of_epoch) = data.next_train()

      # word count
      valid_words += y_count - hparams.batch_size

      logits = model.forward(
        x_valid, x_mask, x_pos_emb_indices,
        y_valid[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous())
      logits = logits.view(-1, hparams.vocab_size)
      labels = y_valid[:, 1:].contiguous().view(-1)
      valid_loss += crit(logits, labels).sum().data[0]

      if end_of_epoch:
        break

    curr_time = time.time()
    log_string = "epoch={0:<3d}".format(epoch + 1)
    log_string += " mins={0:<.2f}".format((curr_time - start_time) / 60.0)
    log_string += " val_ppl={0:<.2f}".format(np.exp(valid_loss / valid_words))
    print(log_string)


def main():
  print("-" * 80)
  if not os.path.isdir(args.output_dir):
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
  elif args.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(args.output_dir))
    shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

  print("-" * 80)
  log_file = os.path.join(args.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  train()

if __name__ == "__main__":
  main()
