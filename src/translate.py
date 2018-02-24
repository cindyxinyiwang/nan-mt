import argparse
import _pickle as pickle
import shutil
import gc
import os
import sys
import time

import numpy as np

from data_utils import DataLoader
from hparams import *
from utils import *
from models import *

import torch
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="Neural MT translator")

add_argument(parser, "model_dir", type="str", default="outputs",
             help="root directory of saved model")
add_argument(parser, "src_file", type="str", default="data/src",
             help="src file to translate")
add_argument(parser, "out_dir", type="str", default="outputs",
             help="output dir of translation")
add_argument(parser, "beam_size", type="int", default="5",
             help="beam size")
add_argument(parser, "max_len", type="int", default="50",
             help="max len")

args = parser.parse_args()

model_file_name = os.path.join(args.output_dir, "model.pt")
model = torch.load(model_file_name)

hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
hparams = torch.load(hparams_file_name)

data = DataLoader(hparams=hparams)

while True:
  ((x_train, x_mask, x_pos_emb_indices, x_count),
  (y_train, y_mask, y_pos_emb_indices, y_count),
  end_of_epoch) = data.next_train()

  all_hyps, all_scores = model.translate_batch(x_train, x_mask, x_pos_emb_indices, 
  	                                            args.beam_size, args.max_len)

  
