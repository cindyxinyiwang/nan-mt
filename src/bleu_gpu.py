import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import *

import numpy as np
import os

class NgramConv(nn.Module):
  def __init__(self, n, cuda=False):
    super(NgramConv, self).__init__()
    self.cov = nn.Conv2d(1, 1, n)
    kernel = torch.zeros(n, n)
    nn.init.eye(kernel)
    self.cov.weight = nn.Parameter(kernel.view(1, 1, n, n), requires_grad=False)
    self.cov.bias = nn.Parameter(torch.zeros(1), requires_grad=False)
    if cuda:
      self.cov = self.cov.cuda()
  def forward(self, x):
    return self.cov(x)

def match(M_hr, M_hh, n, ngram_cov, cuda=False):
  """
  M_hr: [batch_size, hyp_len, ref_len]
  M_hh: [batch_size, hyp_len, hyp_len]
  number of matching unigram
  return: m [batch_size, 1]
  """
  batch_size, hyp_len, ref_len = M_hr.size()
  if n > 1:
    M_hr_v = Variable(M_hr.unsqueeze(1).float(), requires_grad=False)
    M_hh_v = Variable(M_hh.unsqueeze(1).float(), requires_grad=False)
    M_hr_n = torch.eq(ngram_cov.forward(M_hr_v).squeeze(1), n).data.float()
    M_hh_n = torch.eq(ngram_cov.forward(M_hh_v).squeeze(1), n).data.float()
  else:
    M_hr_n = M_hr.float()
    M_hh_n = M_hh.float()
  # [batch_size, hyp_len]
  o_hyp = M_hh_n.sum(dim=2)
  o_ref = M_hr_n.sum(dim=2)
  #print("o_hyp", o_hyp)
  #print("o_ref", o_ref)
  zero_mask = torch.eq(o_hyp, 0.)
  o_hyp = o_hyp.masked_fill_(zero_mask, 1)
  div = torch.div(o_ref, o_hyp)
  #print("div", div)
  one = torch.FloatTensor([1.])
  if cuda: 
    one = one.cuda()
  m = torch.min(div, one)
  m = m.masked_fill_(zero_mask, 0)
  m = m.sum(dim=1)
  return m

def bleu(hyp, ref, hyp_mask, ref_mask, ngram_cov_list, cuda=False):
  """
  hyp: [batch_size, hyp_len], word index of hypothesis 
  ref: [batch_size, ref_len], word index of reference
  """
  batch_size, hyp_len = hyp.size()
  #print(batch_size)
  batch_size, ref_len = ref.size()
  #print(batch_size)
  M_hh = torch.eq(hyp.unsqueeze(2), hyp.unsqueeze(1)).long()
  M_hr = torch.eq(hyp.unsqueeze(2), ref.unsqueeze(1)).long()
  M_hr = M_hr.masked_fill_(hyp_mask.unsqueeze(2), 0)
  M_hr = M_hr.masked_fill_(ref_mask.unsqueeze(1), 0)
  M_hh = M_hh.masked_fill_(hyp_mask.unsqueeze(2), 0)
  M_hh = M_hh.masked_fill_(hyp_mask.unsqueeze(1), 0)
  result = 0.
  inv_hyp_mask = 1 - hyp_mask
  inv_ref_mask = 1 - ref_mask
  for i in range(1, 5):
    match_i = match(M_hr, M_hh, i, ngram_cov_list[i-1], cuda)
    l_hi = inv_hyp_mask.int().sum() - batch_size * i + batch_size
    prec_i = match_i.sum() / l_hi
    #print(i, "match_i", match_i)
    #print(i, "correct: {}".format(match_i.sum()), "total: {}".format(l_hi), prec_i)
    result += np.log(prec_i) 
  bp = min(1., np.exp(1. - inv_ref_mask.int().sum()/inv_hyp_mask.int().sum()))
  #print(bp)
  return bp * np.exp(result/4)

def main(output_dir, hyp_file, ref_file):
  hparams_file_name = os.path.join(output_dir, "hparams.pt")

  hparams = torch.load(hparams_file_name)
  hparams.source_test = hyp_file
  hparams.target_test = ref_file
  hparams.batch_size = 32
  hparams.source_vocab = hparams.target_vocab = "vocab.de"
  hparams.data_path = "data/bleu/"
  hparams.cuda = False
  data = DataLoader(hparams=hparams, decode=True)

  h, r, _, _= data.next_test(32)
  hyp, hyp_mask, _, _ = h
  ref, ref_mask, _, _ = r
  extra_mask = torch.eq(hyp.data, hparams.bos_id)
  hyp_mask = hyp_mask | extra_mask
  extra_mask = torch.eq(hyp.data, hparams.eos_id)
  hyp_mask = hyp_mask | extra_mask
  extra_mask = torch.eq(ref.data, hparams.bos_id)
  ref_mask = ref_mask | extra_mask
  extra_mask = torch.eq(ref.data, hparams.eos_id)
  ref_mask = ref_mask | extra_mask

  ngram_cov_list = [
    None,
    NgramConv(2, hparams.cuda),
    NgramConv(3, hparams.cuda),
    NgramConv(4, hparams.cuda)]
  print(
    "bleu score: ",
    bleu(hyp.data, ref.data, hyp_mask, ref_mask, ngram_cov_list, hparams.cuda))

if __name__ == "__main__":
  #hyp = torch.LongTensor( [[1, 4, 5, 3, 2, 3], [2, 4, 5, 3, 0, 0]] )
  #ref = torch.LongTensor( [[2, 4, 5, 3, 0, 0], [1, 4, 5, 3, 2, 3]] )
  #hyp_mask = torch.ByteTensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]])
  #ref_mask = torch.ByteTensor([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]])
  #ngram_cov_list = [None, NgramConv(2), NgramConv(3), NgramConv(4)]
  #b = bleu(hyp, ref, hyp_mask, ref_mask, ngram_cov_list)
  #print(b)
  main("outputs_exp6_v1", "trans-head32", "ref-head32")

