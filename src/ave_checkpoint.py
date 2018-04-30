from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
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
add_argument(parser, "model_prefix", type="str", default="checkpoint",
             help="root directory of saved model")
add_argument(parser, "model_out", type="str", default="model",
             help="Whether to average model checkpoint")
add_argument(parser, "cuda", type="bool", default=True,
             help="Whether to average model checkpoint")

args = parser.parse_args()
model_files = [fn for fn in os.listdir(args.model_dir) 
        if args.model_prefix in fn]
model_list = []
for model_file_name in model_files:
  model_file_name = os.path.join(args.model_dir, model_file_name)
  print("loading model {}".format(model_file_name))
  if not args.cuda:
    model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
  else:
    model = torch.load(model_file_name)
  model.eval()
  model_list.append(model)
model = model_list[0]
for name, param in model.named_parameters():
  for i in range(1, len(model_list)):
    n_p = Variable(model_list[i].state_dict().get(name)) 
    param = param + n_p 
  model.state_dict()[name] = param / len(model_list)
print("saving file to {}".format(args.model_out))
torch.save(model, os.path.join(args.model_dir, args.model_out))
