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

  # TODO(hyhieu,cindyxinyiwang): build models here!
  model = Transformer(hparams=hparams)

  # train loop
  print("-" * 80)
  print("Start training")
  for epoch in range(hparams.num_epochs):
    start_time = time.time()
    step = 0
    target_words = 0

    # training activities
    while True:
      ((x_train, x_mask, x_pos_emb_indices),
       (y_train, y_mask, y_pos_emb_indices), end_of_epoch) = data.next_train()
      target_words += np.sum(y_mask.cpu().numpy())

      logits = model.forward(x_train, x_mask, x_pos_emb_indices,
                             y_train, y_mask, y_pos_emb_indices)

      step += 1
      if step % args.log_every == 0:
        curr_time = time.time()
        elapsed = (curr_time - start_time) / 60.0
        log_string = "step={0:<5d}".format(step)
        log_string += " mins={0:<5.2f}".format(elapsed)
        log_string += " wpm={0:<5.2f}".format(target_words / elapsed)
        print(log_string)

      if end_of_epoch:
        break

    # End-of-Epoch activites, e.g: compute PPL, BLEU, etc.
    while True:
      ((x_valid, x_mask, x_pos_emb_indices),
       (y_valid, y_mask, y_pos_emb_indices), end_of_epoch) = data.next_train()

      # TODO(hyhieu,cindyxinyiwang): Beam search, BLEU, PPL, etc.

      if end_of_epoch:
        break

    curr_time = time.time()
    log_string = "epoch={0:<3d}".format(epoch + 1)
    log_string += " mins={0:<.2f}".format((curr_time - start_time) / 60.0)
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
