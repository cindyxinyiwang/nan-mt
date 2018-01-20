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

from utils import *

parser = argparse.ArgumentParser(description="Neural MT")

parser.add_argument("--reset_output_dir", type=bool, default="reset_output_dir",
                    help="delete output directory if it exists")
parser.add_argument("--output_dir", type=str, default="outputs",
                    help="path to output directory")
parser.add_argument("--data_dir", type=str, default="data",
                    help="path to data directory")
parser.add_argument("--data_file", type=str, default="data.pkl",
                    help="name of pre-processed file")

args = parser.parse_args()


def train():
  raise NotImplementedError("Bite me!")


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

  print("-" * 80)
  data_file = os.path.join(args.data_dir, args.data_file)
  print("Reading data from file {}".format(data_file))

  train()

if __name__ == "__main__":
  main()
