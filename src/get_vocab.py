from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import shutil
import sys
import time

import numpy as np

THRESHOLD = 3  # words that appear less than THRESHOLD times are <unk>
DATA_PATH = "data/raw/de-en"
INP_NAMES = ["train.en", "train.de"]
OUT_NAMES = ["vocab.en", "vocab.de"]

def main():
  for inp_name, out_name in zip(INP_NAMES, OUT_NAMES):
    inp_file_name = os.path.join(DATA_PATH, inp_name)
    with open(inp_file_name) as finp:
      lines = finp.read().split("\n")

    vocab = {
      "<pad>": 0,
      "<unk>": 1,
      "<s>": 2,
      "</s>": 3,
    }
    word_counts = {}
    num_lines = 0
    for line in lines:
      line = line.strip()
      if not line:
        continue
      tokens = line.split(" ")
      for token in tokens:
        if token not in vocab:
          index = len(vocab)
          vocab[token] = index
          word_counts[token] = 1
        else:
          word_counts[token] += 1
      num_lines += 1
      if num_lines % 50000 == 0:
        print("Read {0:>6d} lines".format(num_lines))
        sys.stdout.flush()
    print("Read {0:>6d} lines. vocab_size={1}".format(num_lines, len(vocab)))
    sys.stdout.flush()

    log_string = ""
    for word, idx in vocab.items():
      if word in word_counts and word_counts[word] < THRESHOLD:
        continue
      log_string += "{0}▁{1}\n".format(word, idx)

    out_name = os.path.join(DATA_PATH, out_name)
    print("Saving vocab to '{0}'".format(out_name))
    sys.stdout.flush()
    with open(out_name, "w") as fout:
      fout.write(log_string)

if __name__ == "__main__":
  main()
