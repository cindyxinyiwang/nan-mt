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
import re

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataLoader
from hparams import *
from utils import *
from models import *

parser = argparse.ArgumentParser(description="Neural MT")

add_argument(parser, "load_model", type="bool", default=False, help="load an existing model")
add_argument(parser, "reset_output_dir", type="bool", default=False, help="delete output directory if it exists")
add_argument(parser, "output_dir", type="str", default="outputs", help="path to output directory")
add_argument(parser, "log_every", type="int", default=50, help="how many steps to write log")
add_argument(parser, "eval_every", type="int", default=500, help="how many steps to compute valid ppl")
add_argument(parser, "clean_mem_every", type="int", default=10, help="how many steps to clean memory")
add_argument(parser, "eval_bleu", type="bool", default=False, help="if calculate BLEU score for dev set")
add_argument(parser, "beam_size", type="int", default=5, help="beam size for dev BLEU")

add_argument(parser, "cuda", type="bool", default=False, help="GPU or not")

add_argument(parser, "max_len", type="int", default=300, help="maximum len considered on the target side")
add_argument(parser, "n_train_sents", type="int", default=None, help="max number of training sentences to load")

add_argument(parser, "data_path", type="str", default=None, help="path to all data")
add_argument(parser, "source_train", type="str", default=None, help="source train file")
add_argument(parser, "target_train", type="str", default=None, help="target train file")
add_argument(parser, "source_valid", type="str", default=None, help="source valid file")
add_argument(parser, "target_valid", type="str", default=None, help="target valid file")
add_argument(parser, "source_vocab", type="str", default=None, help="source vocab file")
add_argument(parser, "target_vocab", type="str", default=None, help="target vocab file")
add_argument(parser, "source_test", type="str", default=None, help="source test file")
add_argument(parser, "target_test", type="str", default=None, help="target test file")

add_argument(parser, "d_word_vec", type="int", default=288, help="size of word and positional embeddings")
add_argument(parser, "d_model", type="int", default=288, help="size of hidden states")
add_argument(parser, "d_inner", type="int", default=512, help="hidden dimension of the position-wise ff")
add_argument(parser, "d_k", type="int", default=64, help="dim of attn keys")
add_argument(parser, "d_v", type="int", default=64, help="dim of attn values")
add_argument(parser, "n_layers", type="int", default=5, help="number of layers in a Transformer stack")
add_argument(parser, "n_heads", type="int", default=2 , help="number of attention heads")
add_argument(parser, "batch_size", type="int", default=32, help="")
add_argument(parser, "n_train_steps", type="int", default=100000, help="")
add_argument(parser, "n_warm_ups", type="int", default=750, help="")

add_argument(parser, "share_emb_and_softmax", type="bool", default=True, help="share embedding and softmax")

add_argument(parser, "dropout", type="float", default=0.1, help="probability of dropping")
add_argument(parser, "label_smoothing", type="float", default=None, help="")
add_argument(parser, "grad_bound", type="float", default=None, help="L2 norm")
add_argument(parser, "init_range", type="float", default=0.1, help="L2 norm")
add_argument(parser, "lr", type="float", default=20.0, help="initial lr")
add_argument(parser, "lr_dec", type="float", default=2.0, help="decrease lr when val_ppl does not improve")


add_argument(parser, "patience", type="int", default=-1,
             help="how many more steps to take before stop. Ignore n_train_stop if patience is set")

args = parser.parse_args()


def eval(model, data, crit, step, hparams, valid_batch_size=20):
  print("Eval at step {0}. valid_batch_size={1}".format(step, valid_batch_size))
  model.eval()
  data.reset_valid()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  valid_bleu = None
  if args.eval_bleu:
    valid_hyp_file = os.path.join(args.output_dir, "dev.trans_{0}".format(step))
    out_file = open(valid_hyp_file, 'w', encoding='utf-8')
  while True:
    # clear GPU memory
    gc.collect()

    # next batch
    ((x_valid, x_mask, x_pos_emb_indices, x_count),
     (y_valid, y_mask, y_pos_emb_indices, y_count),
     batch_size, end_of_epoch) = data.next_valid(valid_batch_size=valid_batch_size)

    # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
    y_count -= batch_size

    # word count
    valid_words += y_count

    logits = model.forward(
      x_valid, x_mask, x_pos_emb_indices,
      y_valid[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous(),
      label_smoothing=False)
    logits = logits.view(-1, hparams.target_vocab_size)
    n_batches += 1
    # if n_batches >= 1:
    #   print(logits[5])
    #   return
    labels = y_valid[:, 1:].contiguous().view(-1)

    val_loss, val_acc = get_performance(crit, logits, labels, hparams)
    valid_loss += val_loss.data[0]
    valid_acc += val_acc.data[0]
    # print("{0:<5d} / {1:<5d}".format(val_acc.data[0], y_count))

    # BLEU eval
    if args.eval_bleu:
      all_hyps, all_scores = model.translate_batch(
        x_valid, x_mask, x_pos_emb_indices, args.beam_size, args.max_len)
      filtered_tokens = set([hparams.bos_id, hparams.eos_id])
      for h in all_hyps:
        h_best = h[0]
        h_best_words = map(lambda wi: data.target_index_to_word[wi],
                           filter(lambda wi: wi not in filtered_tokens, h_best))
        line = ''.join(h_best_words)
        line = line.replace('▁', ' ').strip()
        out_file.write(line + '\n')
    if end_of_epoch:
      break
  val_ppl = np.exp(valid_loss / valid_words)
  log_string = "val_step={0:<6d}".format(step)
  log_string += " loss={0:<6.2f}".format(valid_loss / valid_words)
  log_string += " acc={0:<5.4f}".format(valid_acc / valid_words)
  log_string += " val_ppl={0:<.2f}".format(val_ppl)
  if args.eval_bleu:
    out_file.close()
    ref_file = os.path.join(hparams.data_path, hparams.target_valid)
    bleu_str = subprocess.getoutput("./multi-bleu.perl -lc {0} < {1}".format(ref_file, valid_hyp_file))
    log_string += " {}".format(bleu_str)
  print(log_string)
  model.train()
  return val_ppl


def train():
  if args.load_model:
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
  else:
    hparams = Iwslt16EnDeBpe32SharedParams(
      data_path=args.data_path,
      source_train=args.source_train,
      target_train=args.target_train,
      source_valid=args.source_valid,
      target_valid=args.target_valid,
      source_vocab=args.source_vocab,
      target_vocab=args.target_vocab,
      source_test=args.source_test,
      target_test=args.target_test,
      max_len=args.max_len,
      n_train_sents=args.n_train_sents,
      cuda=args.cuda,
      d_word_vec=args.d_word_vec,
      d_model=args.d_model,
      d_inner=args.d_inner,
      d_k=args.d_k,
      d_v=args.d_v,
      n_layers=args.n_layers,
      n_heads=args.n_heads,
      batch_size=args.batch_size,
      n_train_steps=args.n_train_steps,
      n_warm_ups=args.n_warm_ups,
      share_emb_and_softmax=args.share_emb_and_softmax,
      dropout=args.dropout,
      label_smoothing=args.label_smoothing,
      grad_bound=args.grad_bound,
      init_range=args.init_range,
      lr=args.lr,
      lr_dec=args.lr_dec
    )
  data = DataLoader(hparams=hparams)
  hparams.add_param("source_vocab_size", data.source_vocab_size)
  hparams.add_param("target_vocab_size", data.target_vocab_size)
  hparams.add_param("pad_id", data.pad_id)
  hparams.add_param("unk_id", data.unk_id)
  hparams.add_param("bos_id", data.bos_id)
  hparams.add_param("eos_id", data.eos_id)

  # build or load model model
  print("-" * 80)
  print("Creating model")
  if args.load_model:
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)
  else:
    model = Transformer(hparams=hparams)
  crit = get_criterion(hparams)

  trainable_params = [p for p in model.trainable_parameters()]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  # build or load optimizer
  optim = torch.optim.SGD(trainable_params, lr=hparams.lr)
  if args.load_model:
    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optim from '{0}'".format(optim_file_name))
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

  try:
    extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_val_ppl, lr = torch.load(extra_file_name)
  except:
    step = 0
    best_val_ppl = hparams.target_vocab_size
    lr = args.lr

  # train loop
  print("-" * 80)
  print("Start training")
  start_time = time.time()
  actual_start_time = time.time()
  target_words = 0
  total_loss = 0
  total_corrects = 0
  n_train_batches = data.train_size // hparams.batch_size

  while True:
    # training activities
    model.train()
    while True:
      # next batch
      ((x_train, x_mask, x_pos_emb_indices, x_count),
       (y_train, y_mask, y_pos_emb_indices, y_count),
       batch_size) = data.next_train()

      # book keeping count
      # Since you are shifting y_train, i.e. y_train[:, :-1] and y_train[:, 1:]
      y_count -= batch_size  
      target_words += y_count

      # forward pass
      optim.zero_grad()
      logits = model.forward(
        x_train, x_mask, x_pos_emb_indices,
        y_train[:, :-1], y_mask[:, :-1].contiguous(), y_pos_emb_indices[:, :-1].contiguous())
      logits = logits.view(-1, hparams.target_vocab_size)
      labels = y_train[:, 1:].contiguous().view(-1)
      tr_loss, tr_acc = get_performance(crit, logits, labels, hparams)
      total_loss += tr_loss.data[0]
      total_corrects += tr_acc.data[0]
      tr_loss = tr_loss.div(batch_size)

      # set learning rate
      if step < hparams.n_warm_ups:
        lr = hparams.lr * (step + 1) / hparams.n_warm_ups
      set_lr(optim, lr)

      tr_loss.backward()
      grad_norm = grad_clip(trainable_params, grad_bound=hparams.grad_bound)
      optim.step()

      step += 1
      if step % args.log_every == 0:
        curr_time = time.time()
        elapsed = (curr_time - actual_start_time) / 60.0
        epoch = step // n_train_batches
        log_string = "ep={0:<3d}".format(epoch)
        log_string += " steps={0:<6.2f}".format(step / 1000)
        log_string += " lr={0:<6.3f}".format(lr)
        log_string += " loss={0:<7.2f}".format(tr_loss.data[0])
        log_string += " |g|={0:<6.2f}".format(grad_norm)
        log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
        log_string += " acc={0:<5.4f}".format(total_corrects / target_words)
        log_string += " wpm(K)={0:<5.2f}".format(
          target_words / (1000 * elapsed))
        log_string += " mins={0:<5.2f}".format(elapsed)
        print(log_string)

      # clean up GPU memory
      if step % args.clean_mem_every == 0:
        gc.collect()

      # eval
      if step % args.eval_every == 0:
        val_ppl = eval(model, data, crit, step, hparams, valid_batch_size=30)
        if val_ppl < best_val_ppl:
          best_val_ppl = val_ppl
          save_checkpoint([step, best_val_ppl, lr], model, optim, hparams,
                          args.output_dir)
        else:
          lr /= hparams.lr_dec
        actual_start_time = time.time()
        target_words = 0
        total_loss = 0
        total_corrects = 0

      if step >= hparams.n_train_steps:
        break


    # stop if trained for more than n_train_steps
    if step >= hparams.n_train_steps:
      print("Reach {0} steps. Stop training".format(step))
      val_ppl = eval(model, data, crit, step, hparams, valid_batch_size=30)
      if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        save_checkpoint([step, best_val_ppl, lr], model, optim, hparams,
                        args.output_dir)
      break

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
