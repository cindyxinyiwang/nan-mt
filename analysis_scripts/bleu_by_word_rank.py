import numpy as np
from collections import defaultdict
import math

def merge_dict(d1, d2):
  result = d1
  for key in d2:
    value = d2[key]
    if key in result:
      result[key] = max(result[key], value)
    else:
      result[key] = value
  return result

def sentence2dict(sentence, n):
  words = sentence.split(' ')
  result = {}
  for n in range(1, n + 1):
    for pos in range(len(words) - n + 1):
      gram = ' '.join(words[pos : pos + n])
      if gram in result:
        result[gram] += 1
      else:
        result[gram] = 1
  return result

def bleu(hypo_sen, refs_sen, n):
  correctgram_count = [0] * n
  ngram_count = [0] * n
  #hypo_sen = hypo_c.split('\n')
  #refs_sen = [refs_c[i].split('\n') for i in range(len(refs_c))]
  hypo_length = 0
  ref_length = 0
  
  for num in range(len(hypo_sen)):
    hypo = hypo_sen[num]
    h_length = len(hypo.split(' '))
    hypo_length += h_length
    
    refs = [refs_sen[i][num] for i in range(len(refs_sen))]
    ref_lengths = sorted([len(refs[i].split(' ')) for i in range(len(refs))])
    ref_distances = [abs(r - h_length) for r in ref_lengths]
    
    ref_length += ref_lengths[np.argmin(ref_distances)]
    refs_dict = {}
    for i in range(len(refs)):
      ref = refs[i]
      ref_dict = sentence2dict(ref, n)
      refs_dict = merge_dict(refs_dict, ref_dict)
    
    hypo_dict = sentence2dict(hypo, n)
    
    for key in hypo_dict:
      value = hypo_dict[key]
      length = len(key.split(' '))
      ngram_count[length - 1] += value
      if key in refs_dict:
        correctgram_count[length - 1] += min(value, refs_dict[key])
     
  result = 0.
  bleu_n = [0.] * n
  if correctgram_count[0] == 0:
    return 0.
  for i in range(n):
    if correctgram_count[i] == 0:
      correctgram_count[i] += 1
      ngram_count[i] += 1
    bleu_n[i] = correctgram_count[i] * 1. / ngram_count[i]
    result += math.log(bleu_n[i]) / n
  bp = 1
  if hypo_length < ref_length:
    bp = math.exp(1 - ref_length * 1.0 / hypo_length)
  return bp * math.exp(result)

def get_vocab_rank(filename):
  vocab = defaultdict(int)
  with open(filename, 'r', encoding='utf-8') as myfile:
    for line in myfile:
      toks = line.split()
      for t in toks:
        vocab[t] += 1
  v2r = {}
  r = 1
  for w in sorted(vocab, key=vocab.get, reverse=True):
    v2r[w] = r
    r += 1
  return v2r

def get_sent_rank(train_file, test_file):
  v2r = get_vocab_rank(train_file)
  id2rank = []
  with open(test_file, 'r', encoding='utf-8') as myfile:
    for line in myfile:
      toks = line.split()
      rank = 0
      for t in toks:
        if t in v2r:
          rank += v2r[t]
        else:
          rank += len(v2r)
      id2rank.append(rank / len(toks))
  return id2rank

def group_by_rank(train_src, test_src, test_ref, hyp1, hyp2):
  id2rank = get_sent_rank(train_src, test_src)
  ref_lines = open(test_ref, 'r', encoding='utf-8').readlines()
  hyp1_lines = open(hyp1, 'r', encoding='utf-8').readlines()
  hyp2_lines = open(hyp2, 'r', encoding='utf-8').readlines()

  id2rank = np.array(id2rank)
  rank_range = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250]
  hyp1_bleu = []
  hyp2_bleu = []
  for i in range(len(rank_range)):
    if i == 0:
      left = min(id2rank)
      right = rank_range[i]
      ids = np.where(id2rank<right)
    elif i == len(rank_range) -1:
      left = rank_range[i]
      right = max(id2rank)
      ids = np.where(id2rank>=left) 
    else:
      left, right = rank_range[i], rank_range[i+1]
      ids = np.where(np.logical_and(id2rank>=left, id2rank<right))
    ids = ids[0]
    print("range={} {}, count={}".format(left, right, len(ids)))
    print(ref_lines[ids[0]])
    cur_refs = [ref_lines[i] for i in ids]
    cur_hyp1s = [hyp1_lines[i] for i in ids]
    cur_hyp2s = [hyp2_lines[i] for i in ids]
    b1 = bleu(cur_hyp1s, [cur_refs], 4)
    b2 = bleu(cur_hyp2s, [cur_refs], 4)
    hyp1_bleu.append(b1)
    hyp2_bleu.append(b2)
  return hyp1_bleu, hyp2_bleu

if __name__ == "__main__":
  train_src = "train.de"
  test_src = "test.de"
  test_ref = "test.en"
  hyp1 = "test.baseline"
  hyp2 = "test.source_corruption"
  hyp1_bleu, hyp2_bleu = group_by_rank(train_src, test_src, test_ref, hyp1, hyp2)
  print(hyp1_bleu)
  print(hyp2_bleu)
