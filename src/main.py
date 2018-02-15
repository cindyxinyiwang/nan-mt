from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import _pickle as pickle
import shutil
import gc
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
                    help="how many steps to write log")
parser.add_argument("--clean_mem_every", type=int, default="10",
                    help="how many steps to clean memory")


args = parser.parse_args()


def train():
  # hparams = Iwslt16EnDeTinyParams()
  # data = DataLoader(hparams="tiny")

  hparams = Iwslt16EnDeBpe16Params()
  data = DataLoader(hparams="bpe16")

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
    total_sents = 0

    # training activities
    model.train()
    while True:
      # next batch
      ((x_train, x_mask, x_pos_emb_indices, x_count),
       (y_train, y_mask, y_pos_emb_indices, y_count),
       end_of_epoch) = data.next_train()

      # book keeping count
      target_words += y_count
      total_sents += x_train.size()[0]

      # forward pass
      optim.zero_grad()
      logits = model.forward(
        x_train, x_mask, x_pos_emb_indices,
        y_train[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous())
      logits = logits.view(-1, hparams.vocab_size)
      labels = y_train[:, 1:].contiguous().view(-1)
      tr_loss, tr_acc = get_performance(crit, logits, labels)
      tr_loss = tr_loss.div(hparams.batch_size)
      tr_ppl = np.exp(tr_loss.data[0] * hparams.batch_size / y_count)

      # backward pass and training
      tr_loss.backward()
      optim.step()

      step += 1
      if step % args.log_every == 0:
        curr_time = time.time()
        elapsed = (curr_time - start_time) / 60.0
        log_string = "step={0:<5d}".format(step)
        log_string += " sents={0:<7d}".format(total_sents)
        log_string += " mins={0:<5.2f}".format(elapsed)
        log_string += " ppl={0:<7.2f}".format(tr_ppl)
        log_string += " acc={0:<5.4f}".format(tr_acc.data[0]/ y_count)
        log_string += " wpm={0:<5.2f}".format(target_words / elapsed)
        print(log_string)

      # clean up GPU memory
      if step % args.clean_mem_every == 0:
        gc.collect()

      if end_of_epoch:
        break

    # End-of-Epoch activites, e.g: compute PPL, BLEU, etc.
    # Save checkpoint
    print("-" * 80)
    save_checkpoint(model, optim, args.output_dir)

    # Eval
    model.eval()
    valid_words = 0
    valid_loss = 0
    valid_acc = 0
    while True:
      # clear GPU memory
      gc.collect()

      # next batch
      ((x_valid, x_mask, x_pos_emb_indices, x_count),
       (y_valid, y_mask, y_pos_emb_indices, y_count),
       end_of_epoch) = data.next_valid()

      # word count
      valid_words += y_count

      logits = model.forward(
        x_valid, x_mask, x_pos_emb_indices,
        y_valid[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous())
      logits = logits.view(-1, hparams.vocab_size)
      labels = y_valid[:, 1:].contiguous().view(-1)
      val_loss, val_acc = get_performance(crit, logits, labels)
      valid_loss += val_loss.data[0]
      valid_acc += val_acc.data[0]

      if end_of_epoch:
        break

    curr_time = time.time()
    log_string = "epoch={0:<3d}".format(epoch + 1)
    log_string += " mins={0:<.2f}".format((curr_time - start_time) / 60.0)
    log_string += "\nvalid:"
    log_string += " loss={0:<6.2f}".format(valid_loss / valid_words)
    log_string += " acc={0:<5.4f}".format(valid_acc / valid_words)
    log_string += " ppl={0:<.2f}".format(np.exp(valid_loss / valid_words))
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
