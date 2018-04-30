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
import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataLoader
from hparams import *
from utils import *
from models import *

parser = argparse.ArgumentParser(description="Neural MT")

add_argument(parser, "cuda", type="bool", default=False,
             help="GPU or not")
add_argument(parser, "load_model", type="bool", default=False,
             help="load an existing model")
add_argument(parser, "reset_output_dir", type="bool", default=False,
             help="delete output directory if it exists")
add_argument(parser, "output_dir", type="str", default="outputs",
             help="path to output directory")
add_argument(parser, "log_every", type="int", default=50,
             help="how many steps to write log")
add_argument(parser, "eval_every", type="int", default=500,
             help="how many steps to compute valid ppl")
add_argument(parser, "clean_mem_every", type="int", default=10,
             help="how many steps to clean memory")
add_argument(parser, "eval_bleu", type="bool", default=False,
             help="if calculate BLEU score for dev set")
add_argument(parser, "beam_size", type="int", default=5,
             help="beam size for dev BLEU")
add_argument(parser, "ppl_thresh", type="float", default=6.1,
             help="threshold for start evaluating on bleu")
add_argument(parser, "max_len", type="int", default=300,
             help="maximum len considered on the target side")
add_argument(parser, "non_batch_translate", type="bool", default=False,
             help="do not use batched beam search")
add_argument(parser, "n_train_sents", type="int", default=None,
             help="max number of training sentences to load")
add_argument(parser, "data_path", type="str", default=None,
             help="path to all data")
add_argument(parser, "source_train", type="str", default=None,
             help="source train file")
add_argument(parser, "target_train", type="str", default=None,
             help="target train file")
add_argument(parser, "source_valid", type="str", default=None,
             help="source valid file")
add_argument(parser, "target_valid", type="str", default=None,
             help="target valid file")
add_argument(parser, "target_valid_ref", type="str", default=None,
             help="target valid file for reference")
add_argument(parser, "source_vocab", type="str", default=None,
             help="source vocab file")
add_argument(parser, "target_vocab", type="str", default=None,
             help="target vocab file")
add_argument(parser, "source_test", type="str", default=None,
             help="source test file")
add_argument(parser, "target_test", type="str", default=None,
             help="target test file")

add_argument(parser, "d_word_vec", type="int", default=288,
             help="size of word and positional embeddings")
add_argument(parser, "d_model", type="int", default=288,
             help="size of hidden states")
add_argument(parser, "d_inner", type="int", default=512,
             help="hidden dimension of the position-wise ff")
add_argument(parser, "d_k", type="int", default=64,
             help="dim of attn keys")
add_argument(parser, "d_v", type="int", default=64,
             help="dim of attn values")
add_argument(parser, "n_layers", type="int", default=5,
             help="number of layers in a Transformer stack")
add_argument(parser, "n_heads", type="int", default=2 ,
             help="number of attention heads")
add_argument(parser, "batch_size", type="int", default=32,
             help="")
add_argument(parser, "valid_batch_size", type="int", default=20,
             help="")
add_argument(parser, "batcher", type="str", default="sent",
             help=("sent|word. Batch either by number of words or "
                   "number of sentences"))
add_argument(parser, "n_train_steps", type="int", default=100000,
             help="")
add_argument(parser, "n_warm_ups", type="int", default=750,
             help="")
add_argument(parser, "optim_switch", type="int", default=None,
             help="Switch from adam to SGD")
add_argument(parser, "pos_emb_size", type="int", default=None,
             help="Number of positional embedding steps. None means sinusoid")
add_argument(parser, "raml_source", type="bool", default=False,
             help="Sample a corrupted source sentence during training")
add_argument(parser, "raml_target", type="bool", default=False,
             help="Sample a corrupted target sentence during training")
add_argument(parser, "src_pad_corrupt", type="bool", default=False,
             help="Use pad id as token for corrupted source sentence")
add_argument(parser, "trg_pad_corrupt", type="bool", default=False,
             help="Use pad id as token for corrupted target sentence")
add_argument(parser, "raml_tau", type="float", default=1.0,
             help="Temperature parameter of RAML")
add_argument(parser, "raml_src_tau", type="float", default=None,
             help="Temperature parameter of RAML on source")
add_argument(parser, "dist_corrupt", type="bool", default=False,
             help="Use distance of word vector to select corrupted words")
add_argument(parser, "dist_corrupt_tau", type="float", default=20.0,
             help="Temperature parameter of corrupting by distance")
add_argument(parser, "glove_emb_file", type="str", default=None,
             help="Path to the word embedding file for raml corruption")
add_argument(parser, "glove_emb_dim", type="int", default=None,
             help="word embed dimension of the glove emb")
add_argument(parser, "max_glove_vocab_size", type="int", default=None,
             help="maximum number of glove vocab to load")
add_argument(parser, "share_emb_and_softmax", type="bool", default=True,
             help="share embedding and softmax")
add_argument(parser, "dropout", type="float", default=0.1,
             help="probability of dropping")
add_argument(parser, "label_smoothing", type="float", default=None,
             help="")
add_argument(parser, "grad_bound", type="float", default=None,
             help="L2 norm")
add_argument(parser, "init_range", type="float", default=0.1,
             help="L2 norm")
add_argument(parser, "lr_adam", type="float", default=20.0,
             help="initial lr")
add_argument(parser, "lr_sgd", type="float", default=20.0,
             help="initial lr")
add_argument(parser, "lr_dec", type="float", default=2.0,
             help="decrease lr when val_ppl does not improve")
add_argument(parser, "l2_reg", type="float", default=0.0,
             help="L2 weight penalty")
add_argument(parser, "lr_schedule", type="bool", default=False,
             help="enable lr schedule")
add_argument(parser, "optim", type="str", default="sgd",
             help="sgd|adam")
add_argument(parser, "init_type", type="str", default="uniform",
             help=("uniform|xavier_uniform|xavier_normal|"
                   "kaiming_uniform|kaiming_normal"))
add_argument(parser, "loss_norm", type="str", default="sent",
             help=("sent|word. normalize loss in a minibatch by "
                  "number of words or number of sentences"))
add_argument(parser, "patience", type="int", default=-1,
             help=("how many more steps to take before stop. "
                   "Ignore n_train_step if patience is set"))
add_argument(parser, "seed", type="int", default=19920206,
             help="random seed")
add_argument(parser, "save_nbest", type="int", default=1,
             help="save the n best checkpoint")
add_argument(parser, "checkpoint", type="int", default=0,
             help="save the n recent checkpoint")
args = parser.parse_args()

if args.raml_src_tau is None:
  args.raml_src_tau = args.raml_tau

def eval(model, data, crit, step, hparams, eval_bleu=False,
         valid_batch_size=20):
  print("Eval at step {0}. valid_batch_size={1}".format(
    step, args.valid_batch_size))

  model.eval()
  data.reset_valid()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  valid_bleu = None
  if eval_bleu:
    valid_hyp_file = os.path.join(args.output_dir, "dev.trans_{0}".format(step))
    out_file = open(valid_hyp_file, 'w', encoding='utf-8')
  while True:
    # clear GPU memory
    gc.collect()

    # next batch
    ((x_valid, x_mask, x_pos_emb_indices, x_count),
     (y_valid, y_mask, y_pos_emb_indices, y_count),
     batch_size, end_of_epoch) = data.next_valid(
       valid_batch_size=args.valid_batch_size)

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
    # valid_loss += val_loss.data[0]
    valid_loss += val_loss.data[0]
    valid_acc += val_acc.data[0]
    # print("{0:<5d} / {1:<5d}".format(val_acc.data[0], y_count))

    # BLEU eval
    if eval_bleu:
      if args.non_batch_translate:
        all_hyps, all_scores = model.translate(
          x_valid, x_mask, x_pos_emb_indices, args.beam_size, args.max_len)        
      else:
        all_hyps, all_scores = model.translate_batch(
          x_valid, x_mask, x_pos_emb_indices, args.beam_size, args.max_len)
      filtered_tokens = set([hparams.bos_id, hparams.eos_id])
      for h in all_hyps:
        h_best = h[0]
        h_best_words = map(
          lambda wi: data.target_index_to_word[wi],
          filter(lambda wi: wi not in filtered_tokens, h_best))
        line = ' '.join(h_best_words)
        #line = ''.join(h_best_words)
        #line = line.replace('▁', ' ').strip()
        out_file.write(line + '\n')
    if end_of_epoch:
      break
  val_ppl = np.exp(valid_loss / valid_words)
  log_string = "val_step={0:<6d}".format(step)
  log_string += " loss={0:<6.2f}".format(valid_loss / valid_words)
  log_string += " acc={0:<5.4f}".format(valid_acc / valid_words)
  log_string += " val_ppl={0:<.2f}".format(val_ppl)
  if eval_bleu:
    out_file.close()
    if args.target_valid_ref:
      ref_file = args.target_valid_ref
    else:
      ref_file = os.path.join(hparams.data_path, args.target_valid)
    bleu_str = subprocess.getoutput(
      "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file))
    log_string += "\n{}".format(bleu_str)
    bleu_str = bleu_str.split('\n')[-1].strip()
    reg = re.compile("BLEU = ([^,]*).*")
    valid_bleu = float(reg.match(bleu_str).group(1))
  print(log_string)
  model.train()
  return val_ppl, valid_bleu


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
      batcher=args.batcher,
      n_train_steps=args.n_train_steps,
      n_warm_ups=args.n_warm_ups,
      share_emb_and_softmax=args.share_emb_and_softmax,
      dropout=args.dropout,
      label_smoothing=args.label_smoothing,
      grad_bound=args.grad_bound,
      init_range=args.init_range,
      optim_switch=args.optim_switch,
      lr_adam=args.lr_adam,
      lr_sgd=args.lr_sgd,
      lr_dec=args.lr_dec,
      l2_reg=args.l2_reg,
      loss_norm=args.loss_norm,
      init_type=args.init_type,
      pos_emb_size=args.pos_emb_size,
      raml_source=args.raml_source,
      raml_target=args.raml_target,
      raml_tau=args.raml_tau,
      raml_src_tau=args.raml_src_tau,
      src_pad_corrupt=args.src_pad_corrupt,
      trg_pad_corrupt=args.trg_pad_corrupt,
      dist_corrupt=args.dist_corrupt,
      dist_corrupt_tau=args.dist_corrupt_tau,
      glove_emb_file=args.glove_emb_file,
      glove_emb_dim=args.glove_emb_dim,
      max_glove_vocab_size=args.max_glove_vocab_size,
    )
  data = DataLoader(hparams=hparams)
  hparams.add_param("source_vocab_size", data.source_vocab_size)
  hparams.add_param("target_vocab_size", data.target_vocab_size)
  hparams.add_param("pad_id", data.pad_id)
  hparams.add_param("unk_id", data.unk_id)
  hparams.add_param("bos_id", data.bos_id)
  hparams.add_param("eos_id", data.eos_id)
  hparams.add_param("l2_reg", args.l2_reg)
  hparams.add_param("n_train_steps", args.n_train_steps)

  # build or load model model
  print("-" * 80)
  print("Creating model")
  if args.load_model:
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)
  else:
    print("Initialize with {}".format(hparams.init_type))
    model = Transformer(hparams=hparams, init_type=hparams.init_type)
  crit = get_criterion(hparams)

  trainable_params = [
    p for p in model.trainable_parameters() if p.requires_grad]

  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  # build or load optimizer
  if args.optim == 'adam':
    print("Using adam optimizer...")
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr_adam,
                             weight_decay=hparams.l2_reg)
  else:
    print("Using sgd optimizer...")
    optim = torch.optim.SGD(trainable_params, lr=hparams.lr_sgd,
                            weight_decay=hparams.l2_reg)
  print("Using transformer lr schedule: {}".format(args.lr_schedule))

  if args.load_model:
    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optim from '{0}'".format(optim_file_name))
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)
    try:
      extra_file_name = os.path.join(args.output_dir, "extra.pt")
      if args.checkpoint > 0:
        (step,
         best_val_ppl,
         best_val_bleu,
         cur_attempt,
         lr,
         checkpoint_queue) = torch.load(extra_file_name)
      else:
        (step,
         best_val_ppl,
         best_val_bleu,
         cur_attempt,
         lr) = torch.load(extra_file_name)
    except:
      raise RuntimeError("Cannot load checkpoint!")
  else:
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr_adam,
                             weight_decay=hparams.l2_reg)
    step = 0
    best_val_ppl = 1e10
    best_val_bleu = 0
    cur_attempt = 0
    lr = hparams.lr_adam
    if args.checkpoint > 0:
      checkpoint_queue = deque(["checkpoint_" + str(i) for i in range(args.checkpoint)])
  if not type(best_val_ppl) == dict:
    best_val_ppl = {}
    best_val_bleu = {}
    if args.save_nbest > 1:
      for  i in range(args.save_nbest):
        best_val_ppl['model'+str(i)] = 1e10
        best_val_bleu['model'+str(i)] = 0
    else:
      best_val_ppl['model'] = 1e10
      best_val_bleu['model'] = 0
  set_patience = args.patience >= 0
  # train loop
  print("-" * 80)
  print("Start training")
  ppl_thresh = args.ppl_thresh
  start_time = time.time()
  actual_start_time = time.time()
  target_words, total_loss, total_corrects = 0, 0, 0
  n_train_batches = data.n_train_batches

  while True:
    # training activities
    model.train()
    while True:
      # next batch
      if hparams.raml_source and hparams.raml_target:
        ((x_train_raml, x_train, x_mask, x_pos_emb_indices, x_count),
         (y_train_raml, y_train, y_mask, y_pos_emb_indices, y_count),
         batch_size) = data.next_train()
      elif hparams.raml_source:
        ((x_train_raml, x_train, x_mask, x_pos_emb_indices, x_count),
         (y_train, y_mask, y_pos_emb_indices, y_count),
         batch_size) = data.next_train()
      elif hparams.raml_target:
        ((x_train, x_mask, x_pos_emb_indices, x_count),
         (y_train_raml, y_train, y_mask, y_pos_emb_indices, y_count),
         batch_size) = data.next_train()
      else:
        ((x_train, x_mask, x_pos_emb_indices, x_count),
         (y_train, y_mask, y_pos_emb_indices, y_count),
         batch_size) = data.next_train()

      # book keeping count
      # Since you are shifting y_train, i.e. y_train[:, :-1] and y_train[:, 1:]
      y_count -= batch_size  
      target_words += y_count

      # forward pass
      optim.zero_grad()
      if hparams.raml_source and hparams.raml_target:
        logits = model.forward(
          x_train_raml, x_mask, x_pos_emb_indices,
          y_train_raml[:, :-1], y_mask[:, :-1],
          y_pos_emb_indices[:, :-1].contiguous())
      elif hparams.raml_source:
        logits = model.forward(
          x_train_raml, x_mask, x_pos_emb_indices,
          y_train[:, :-1], y_mask[:, :-1],
          y_pos_emb_indices[:, :-1].contiguous())
      elif hparams.raml_target:
        logits = model.forward(
          x_train, x_mask, x_pos_emb_indices,
          y_train_raml[:, :-1], y_mask[:, :-1],
          y_pos_emb_indices[:, :-1].contiguous())
      else:
        logits = model.forward(
          x_train, x_mask, x_pos_emb_indices,
          y_train[:, :-1], y_mask[:, :-1], y_pos_emb_indices[:, :-1].contiguous())
      logits = logits.view(-1, hparams.target_vocab_size)
      if hparams.raml_target:
        labels = y_train_raml[:, 1:].contiguous().view(-1)
      else:
        labels = y_train[:, 1:].contiguous().view(-1)
      tr_loss, tr_acc = get_performance(crit, logits, labels, hparams)
      total_loss += tr_loss.data[0]
      total_corrects += tr_acc.data[0]

      # normalizing tr_loss
      if hparams.loss_norm == "sent":
        loss_div = batch_size
      elif hparams.loss_norm == "word":
        assert y_count == (1 - y_mask[:, 1:].int()).sum()
        loss_div = y_count
      else:
        raise ValueError("Unknown batcher '{0}'".format(hparams.batcher))

      # set learning rate
      if args.lr_schedule:
        s = step + 1
        lr = pow(hparams.d_model, -0.5) * min(
          pow(s, -0.5), s * pow(hparams.n_warm_ups, -1.5))
      else:
        if step < hparams.n_warm_ups:
          if hparams.optim_switch is not None and step < hparams.optim_switch:
            base_lr = hparams.lr_adam 
          else:
            base_lr = hparams.lr_sgd 
          lr = base_lr * (step + 1) / hparams.n_warm_ups

      tr_loss.div_(loss_div).backward()
      set_lr(optim, lr)

      grad_norm = grad_clip(trainable_params, grad_bound=hparams.grad_bound)
      optim.step()

      step += 1
      if step % args.log_every == 0:
        epoch = step // data.n_train_batches
        curr_time = time.time()
        since_start = (curr_time - start_time) / 60.0
        elapsed = (curr_time - actual_start_time) / 60.0
        log_string = "ep={0:<3d}".format(epoch)
        log_string += " steps={0:<6.2f}".format(step / 1000)
        log_string += " lr={0:<9.7f}".format(lr)
        log_string += " loss={0:<7.2f}".format(tr_loss.data[0])
        log_string += " |g|={0:<6.2f}".format(grad_norm)
        log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
        log_string += " acc={0:<5.4f}".format(total_corrects / target_words)
        log_string += " wpm(K)={0:<5.2f}".format(
          target_words / (1000 * elapsed))
        log_string += " mins={0:<5.2f}".format(since_start)
        print(log_string)

      if step == hparams.optim_switch:
        lr = hparams.lr_sgd
        print(("Reached {0} steps. Switching from Adam to SGD "
               "with learning_rate {1:<9.7f}").format(step, lr))
        optim = torch.optim.SGD(trainable_params, lr=hparams.lr_sgd,
                                weight_decay=hparams.l2_reg)

      # clean up GPU memory
      if step % args.clean_mem_every == 0:
        gc.collect()

      # eval
      if step % args.eval_every == 0:
        val_ppl, val_bleu = eval(
          model, data, crit, step, hparams, min(best_val_ppl.values()) < ppl_thresh,
          valid_batch_size=args.valid_batch_size)

        # determine whether to update best_val_ppl or best_val_bleu
        based_on_bleu = args.eval_bleu and (min(best_val_ppl.values()) < ppl_thresh)
        if based_on_bleu:
          if min(best_val_bleu.values()) <= val_bleu:
            save_model_name = min(best_val_bleu, key=best_val_bleu.get)
            best_val_bleu[save_model_name] = val_bleu
            save = True
          else:
            save = False
        else:
          if max(best_val_ppl.values()) >= val_ppl:
            save_model_name = max(best_val_ppl, key=best_val_ppl.get)
            best_val_ppl[save_model_name] = val_ppl
            save = True
          else:
            save = False
        if args.checkpoint > 0:
          cur_name = checkpoint_queue.popleft()
          checkpoint_queue.append(cur_name)
          print("Saving checkpoint to {}".format(cur_name))
          torch.save(model, os.path.join(args.output_dir, cur_name))
        if save:
          save_extra = [step, best_val_ppl, best_val_bleu, cur_attempt, lr]
          if args.checkpoint > 0:
            save_extra.append(cur_name)
          save_checkpoint(save_extra,
                          model, optim, hparams, args.output_dir, save_model_name)
          cur_attempt = 0
        else:
          lr /= hparams.lr_dec
          cur_attempt += 1
        actual_start_time = time.time()
        target_words = 0
        total_loss = 0
        total_corrects = 0

      if set_patience:
        if cur_attempt >= args.patience: break
      else:
        if step >= hparams.n_train_steps: break

    # stop if trained for more than n_train_steps
    stop = False
    if set_patience and cur_attempt >= args.patience: 
      stop = True
    elif not set_patience and step > hparams.n_train_steps:
      stop = True
    if stop:
      print("Reach {0} steps. Stop training".format(step))
      based_on_bleu = args.eval_bleu and (min(best_val_ppl.values()) < ppl_thresh)
      if based_on_bleu:
        if min(best_val_bleu.values()) <= val_bleu:
          save_model_name = min(best_val_bleu, key=best_val_bleu.get)
          best_val_bleu[save_model_name] = val_bleu
          save = True
        else:
          save = False
      else:
        if max(best_val_ppl) >= val_ppl:
          save_model_name = max(best_val_ppl, key=best_val_ppl.get)
          best_val_ppl[save_model_name] = val_ppl
          save = True
        else:
          save = False

      if save:
        save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr],
                        model, optim, hparams, args.output_dir, save_model_name)
      break


def main():
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

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
