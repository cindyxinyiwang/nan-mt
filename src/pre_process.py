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

DATA_PATH = "data/en-de"

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

def _align(x_file, y_file):
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

    x_aligned.append(x_line)
    y_aligned.append(y_line)

  with open(x_file, "w") as fout:
    fout.write("\n".join(x_aligned))

  with open(y_file, "w") as fout:
    fout.write("\n".join(y_aligned))

def main():
  print("-" * 80)
  _strip_tags("train.tags.en-de.en", "train.en")
  _strip_tags("train.tags.en-de.de", "train.de")
  _align("train.en", "train.de")

  print("-" * 80)
  _strip_tags("IWSLT16.TED.dev2010.en-de.de.xml", "dev2010.de") 
  _strip_tags("IWSLT16.TED.dev2010.en-de.en.xml", "dev2010.en") 
  _align("dev2010.en", "dev2010.de") 

  print("-" * 80)
  _strip_tags("IWSLT16.TED.tst2010.en-de.de.xml", "tst2010.de") 
  _strip_tags("IWSLT16.TED.tst2010.en-de.en.xml", "tst2010.en") 
  _align("tst2010.en", "tst2010.de") 

  print("-" * 80)
  _strip_tags("IWSLT16.TED.tst2011.en-de.de.xml", "tst2011.de") 
  _strip_tags("IWSLT16.TED.tst2011.en-de.en.xml", "tst2011.en") 
  _align("tst2011.en", "tst2011.de") 

  print("-" * 80)
  _strip_tags("IWSLT16.TED.tst2012.en-de.de.xml", "tst2012.de") 
  _strip_tags("IWSLT16.TED.tst2012.en-de.en.xml", "tst2012.en") 
  _align("tst2012.en", "tst2012.de") 

  print("-" * 80)
  _strip_tags("IWSLT16.TED.tst2013.en-de.de.xml", "tst2013.de") 
  _strip_tags("IWSLT16.TED.tst2013.en-de.en.xml", "tst2013.en") 
  _align("tst2013.en", "tst2013.de") 

  print("-" * 80)
  _strip_tags("IWSLT16.TED.tst2014.en-de.de.xml", "tst2014.de") 
  _strip_tags("IWSLT16.TED.tst2014.en-de.en.xml", "tst2014.en") 
  _align("tst2014.en", "tst2014.de") 


if __name__ == "__main__":
  main()

