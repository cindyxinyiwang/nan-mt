from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle
import os
import re
import shutil
import sys
import time

import numpy as np

SRC = "en"
TGT = "zh"
DATA_PATH = "data/raw/{0}-{1}".format(SRC, TGT)

def _strip_tags(inp_file, out_file):
  """Remove tags for IWLST data."""

  inp_file = os.path.join(DATA_PATH, inp_file)
  out_file = os.path.join(DATA_PATH, out_file)
  print("Untagging '{0}' into '{1}'".format(inp_file, out_file))

  with open(inp_file) as finp:
    text = finp.read()
  text = re.sub("<[^<]+>", "", text)

  with open(out_file, "w") as fout:
    fout.write(text)

def _align(x_file, y_file, length_limit=None):
  x_file = os.path.join(DATA_PATH, x_file)
  y_file = os.path.join(DATA_PATH, y_file)
  print("Aligning '{0}' into '{1}'".format(x_file, y_file))

  with open(x_file) as finp:
    x_lines = finp.read().split("\n")

  with open(y_file) as finp:
    y_lines = finp.read().split("\n")

  x_aligned, y_aligned = [], []
  for x_line, y_line in zip(x_lines, y_lines):
    x_line = x_line.strip()
    y_line = y_line.strip()

    # if one of the lines is empty, skip both of them
    if not x_line or not y_line:
      continue

    # if one of the lines is an URL, skip both of them
    if x_line.startswith("http") or y_line.startswith("http"):
      continue

    # if two lines are the same, skip both of them
    if x_line == y_line:
      continue

    # if one of the lines is too long, skip both of them
    x_toks = x_line.split(" ")
    y_toks = y_line.split(" ")
    if (length_limit is not None and
        max(len(x_toks), len(y_toks)) > length_limit):
      continue

    x_aligned.append(x_line)
    y_aligned.append(y_line)

  with open(x_file, "w") as fout:
    fout.write("\n".join(x_aligned))

  with open(y_file, "w") as fout:
    fout.write("\n".join(y_aligned))

def main():
  length_limit = 10

  print("-" * 80)
  _strip_tags("train.tags.{0}-{1}.{0}".format(SRC, TGT), "train.{0}".format(SRC))
  _strip_tags("train.tags.{0}-{1}.{1}".format(SRC, TGT), "train.{0}".format(TGT))
  _align("train.{0}".format(SRC), "train.{0}".format(TGT),
         length_limit=length_limit)

  print("-" * 80)
  _strip_tags("IWSLT14.TED.dev2010.{0}-{1}.{0}.xml".format(SRC, TGT),
              "dev2010.{0}".format(SRC)) 
  _strip_tags("IWSLT14.TED.dev2010.{0}-{1}.{1}.xml".format(SRC, TGT),
              "dev2010.{0}".format(TGT)) 
  _align("dev2010.{0}".format(SRC), "dev2010.{0}".format(TGT),
         length_limit=length_limit) 


if __name__ == "__main__":
  main()

