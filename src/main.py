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
import subprocess

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataLoader
from hparams import *
from utils import *
from models import *

parser = argparse.ArgumentParser(description="Neural MT")

add_argument(parser, "train_set", type="str", default="outputs",
             help="[tiny | bpe16 | bpe32]")

add_argument(parser, "load_model", type="bool", default=False,
             help="load an existing model")

add_argument(parser, "reset_output_dir", type="bool", default=False,
             help="delete output directory if it exists")

add_argument(parser, "output_dir", type="str", default="outputs",
             help="path to output directory")

add_argument(parser, "log_every", type="int", default=50,
             help="how many steps to write log")

add_argument(parser, "clean_mem_every", type="int", default=10,
             help="how many steps to clean memory")

add_argument(parser, "eval_bleu", type="bool", default=False,
             help="if calculate BLEU score for dev set")

add_argument(parser, "beam_size", type="int", default=5,
             help="beam size for dev BLEU")

add_argument(parser, "max_len", type="int", default=50,
             help="maximum hypothesis length for dev BLEU")

args = parser.parse_args()


def eval(model, data, crit, epoch, hparams):
  print("Eval at epoch {0}".format(epoch))
  model.eval()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  if args.eval_bleu:
    valid_hyp_file = os.path.join(args.output_dir, "dev.trans")
    out_file = open(valid_hyp_file, 'w')
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
      y_valid[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous(),
      label_smoothing=False)
    logits = logits.view(-1, hparams.vocab_size)
    labels = y_valid[:, 1:].contiguous().view(-1)
    val_loss, val_acc = get_performance(crit, logits, labels)
    valid_loss += val_loss.data[0]
    valid_acc += val_acc.data[0]

    # BLEU eval
    if args.eval_bleu:
      all_hyps, all_scores = model.translate_batch(
        x_valid, x_mask, x_pos_emb_indices, args.beam_size, args.max_len)
      filtered_tokens = set([hparams.bos_id, hparams.eos_id])
      for h in all_hyps:
        h_best = h[0]
        h_best_words = map(lambda wi: data.target_index_to_word[wi],
                           filter(lambda wi: wi not in filtered_tokens, h_best))
        line = ' '.join(h_best_words)
        line = line.replace(' @@', '')
        out_file.write(line + '\n')
    if end_of_epoch:
      break
  val_ppl = np.exp(valid_loss / valid_words)
  log_string = "epoch={0:<3d}".format(epoch + 1)
  log_string += "\nvalid:"
  log_string += " loss={0:<6.2f}".format(valid_loss / valid_words)
  log_string += " acc={0:<5.4f}".format(valid_acc / valid_words)
  log_string += " val_ppl={0:<.2f}".format(val_ppl)
  if args.eval_bleu:
    out_file.close()
    bleu_str = subprocess.getoutput("./multi-bleu.perl -lc {} < {}".format(hparams.source_valid, valid_hyp_file))
    log_string += " {}".format(bleu_str)
  print(log_string)
  return val_ppl


def train():
  if args.load_model:
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
  else:
    if args.train_set == "tiny":
      hparams = Iwslt16EnDeTinyParams()
    elif args.train_set == "bpe16":
      hparams = Iwslt16EnDeBpe16Params()
    elif args.train_set == "bpe32":
      hparams = Iwslt16EnDeBpe32Params()
    elif args.train_set in H_PARAMS_DICT:
      hparams = H_PARAMS_DICT[args.train_set]()
    else:
      raise ValueError("Unknown train_set '{0}'".format(args.train_set))

  if args.train_set == "tiny":
    data = DataLoader(hparams=hparams)
  elif args.train_set == "bpe16":
    data = DataLoader(hparams=hparams)
  elif args.train_set == "bpe32":
    data = DataLoader(hparams=hparams)
  elif args.train_set in H_PARAMS_DICT:
    data = DataLoader(hparams=hparams)
  else:
    raise ValueError("Unknown train_set '{0}'".format(args.train_set))


  # build or load model model
  print("-" * 80)
  print("Creating model")
  if args.load_model:
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)
  else: model = Transformer(hparams=hparams)
  crit = get_criterion(hparams)

  # build or load optimizer
  num_params = count_params(model.trainable_parameters())
  print("Model has {0} params".format(num_params))
  optim = torch.optim.Adam(model.trainable_parameters(),
                           lr=hparams.learning_rate,
                           betas=(0.9, 0.98), eps=1e-09)
  if args.load_model:
    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optim from '{0}'".format(optim_file_name))
    optimizer_state = torch.load(os.path.join(args.output_dir, "optimizer.pt"))
    optim.load_state_dict(optimizer_state)

  # train loop
  print("-" * 80)
  print("Start training")
  step = 0
  best_val_acc = 1e10  # hparams.vocab_size
  for epoch in range(hparams.n_epochs):
    start_time = time.time()
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

      lr = (np.minimum(1.0 / np.sqrt(step + 1),
                       (step + 1) / (np.sqrt(hparams.n_warm_ups) ** 3)) /
            np.sqrt(hparams.d_model))
      set_lr(optim, lr)
      tr_loss.backward()
      optim.step()

      step += 1
      if step % args.log_every == 0:
        curr_time = time.time()
        elapsed = (curr_time - start_time) / 60.0
        log_string = "step={0:<7d}".format(step)
        log_string += " sen(K)={0:<6.2f}".format(total_sents / 1000)
        log_string += " mins={0:<5.2f}".format(elapsed)
        log_string += " lr={0:<8.6f}".format(lr)
        log_string += " ppl={0:<8.2f}".format(tr_ppl)
        log_string += " acc={0:<5.4f}".format(tr_acc.data[0]/ y_count)
        log_string += " wpm(K)={0:<5.2f}".format(
          target_words / (1000 * elapsed))
        print(log_string)

      # clean up GPU memory
      if step % args.clean_mem_every == 0:
        gc.collect()

      if end_of_epoch:
        break

    # stop if trained for more than n_train_steps
    if step >= hparams.n_train_steps:
      print("Reach {0} steps. Stop training".format(step))
      val_acc = eval(model, data, crit, epoch, hparams)
      if val_acc < best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, optim, hparams, args.output_dir)
      break

    # End-of-Epoch activites, e.g: compute PPL, BLEU, etc.
    # Save checkpoint
    print("-" * 80)

    # Eval
    val_acc = eval(model, data, crit, epoch, hparams)
    if val_acc < best_val_acc:
      best_val_acc = val_acc
      save_checkpoint(model, optim, hparams, args.output_dir)

def main():
  if not os.path.isdir(args.output_dir):
    print("-" * 80)
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
  elif args.reset_output_dir:
    print("-" * 80)
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
