from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PositionalEmbedding(nn.Module):
  def __init__(self, hparams):
    super(PositionalEmbedding, self).__init__()

    self.hparams = hparams

    # precompute frequencies
    assert self.hparams.d_word_vec % 2 == 0
    freq = np.arange(start=0, stop=1.0, step=2 / self.hparams.d_word_vec,
                     dtype=np.float32)
    freq = np.power(10000.0, -freq)

    self.freq = Variable(torch.FloatTensor(freq).view(
      [1, 1, self.hparams.d_word_vec // 2]))

    if self.hparams.cuda:
      self.freq = self.freq.cuda()

  def forward(self, pos_emb_indices, mask):
    """Compute positional embeddings.

    Args:
      pos_emb_indices: Tensor of size [batch_size, max_len].
      mask: Tensor of size [batch_size, max_len]. 0 means to ignore.

    Returns:
      pos_emb: Tensor of size [batch_size, max_len, d_word_vec].
    """

    batch_size, max_len = pos_emb_indices.size()
    pos_emb_indices = pos_emb_indices.view([batch_size, max_len, 1])
    mask = mask.view([batch_size, max_len, 1])

    pos_emb_sin = torch.sin(pos_emb_indices / self.freq).view(
      batch_size, max_len, -1, 1)
    pos_emb_cos = torch.cos(pos_emb_indices / self.freq).view(
      batch_size, max_len, -1, 1)
    pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=3).view(
      batch_size, max_len, self.hparams.d_word_vec)
    pos_emb = pos_emb * mask

    return pos_emb

class LayerNormalization(nn.Module):
  def __init__(self, d_hid, eps=1e-3):
    super(LayerNormalization, self).__init__()

    self.eps = eps
    self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
    self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

  def forward(self, z):
    if z.size(1) == 1: return z

    mu = torch.mean(z, keepdim=True, dim=-1)
    sigma = torch.std(z, keepdim=True, dim=-1)
    ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
    ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
    return ln_out

class ScaledDotProdAttn(nn.Module):
  def __init__(self, dim, dropout=0.1):
    super(ScaledDotProdAttn, self).__init__()
    self.temp = np.power(dim, 0.5)
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, q, k, v, attn_mask=None):
    # (n_head*batch_size, len_q, len_k)
    # attn_mask (n_head*batch_size, len_k, 1)
    attn = torch.bmm(q, k.transpose(1, 2)) / self.temp
    if not attn_mask is None:
        attn.data.masked_fill_(attn_mask, -float('inf'))
    size = attn.size()
    assert len(size) > 2
    # doing softmax along dim 1
    attn = self.softmax(attn.view(size[0] * size[1], -1)).view(
      size[0], size[1], -1)

    # (n_head*batch_size, len_q, dim_v); len_k == len_v
    attn = self.dropout(attn)
    output = torch.bmm(attn, v)

    return output

class MultiHeadAttn(nn.Module):
  def __init__(self, n_head, dim, d_k, d_v, dropout=0.1):
    super(MultiHeadAttn, self).__init__()
    # d_k == d_v, key and value are encoder states
    # d_q = d_k, query is the decoder state
    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v

    self.w_q = nn.Parameter(torch.FloatTensor(n_head, dim, d_k))
    self.w_k = nn.Parameter(torch.FloatTensor(n_head, dim, d_k))
    self.w_v = nn.Parameter(torch.FloatTensor(n_head, dim, d_v))

    self.attention = ScaledDotProdAttn(dim)
    # projection of concatenated attn
    self.proj = nn.Linear(n_head*d_v, dim, bias=True)

    self.dropout = nn.Dropout(dropout)
    self.layer_norm = LayerNormalization(dim)

    init.xavier_normal(self.w_q)
    init.xavier_normal(self.w_k)
    init.xavier_normal(self.w_v)
    init.xavier_normal(self.proj.weight)

  def forward(self, q, k, v, attn_mask=None):
    residual = q 
    
    print(q.size())
    batch_size, len_q, dim = q.size()
    batch_size, len_k, dim = k.size()
    batch_size, len_v, dim = v.size()

    # (n_head * (batch_size * len_q) * dim)
    q_batch = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, dim)
    k_batch = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, dim)
    v_batch = v.repeat(self.n_head, 1, 1).view(self.n_head, -1, dim)

    # (n_head * (batch_size * len_q) * d_k) => (n_head*batch_size * len_q * d_k)
    q_batch = torch.bmm(q_batch, self.w_q).view(-1, len_q, self.d_k)
    k_batch = torch.bmm(k_batch, self.w_k).view(-1, len_k, self.d_k)
    v_batch = torch.bmm(v_batch, self.w_v).view(-1, len_v, self.d_v)

    # (n_head*batch_size, len_q, dim_v)
    if not attn_mask is None:
      attn_mask = attn_mask.clone().repeat(self.n_head, 1, 1)
    outputs = self.attention(q_batch, k_batch, v_batch, attn_mask=attn_mask)

    # (n_heads, batch_size, len_q, dim_v)=>(batch_size, len_q, n_head*dim_v)
    outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)

    # (batch_size, len_q, dim)
    outputs = outputs.view(batch_size*len_q, -1)
    outputs = self.proj(outputs).view(batch_size, len_q, -1)
    outputs = self.dropout(outputs)

    return self.layer_norm(outputs + residual)

class PositionwiseFF(nn.Module):
  def __init__(self, d_hid, d_inner, dropout=0.1):
    super(PositionwiseFF, self).__init__()
    self.w_1 = nn.Conv1d(d_hid, d_inner, 1)
    self.w_2 = nn.Conv1d(d_inner, d_hid, 1)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()
    self.layer_norm = LayerNormalization(d_hid)

  def forward(self, x):
    residual = x
    output = self.relu(self.w_1(x.transpose(1, 2)))
    output = self.w_2(output).transpose(2, 1)
    output = self.dropout(output)
    return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
  """Compose multi-head attention and positionwise feeding."""

  def __init__(self, dim, d_inner, n_head, d_k, d_v, dropout=0.1):
    super(EncoderLayer, self).__init__()
    self.attn = MultiHeadAttn(n_head, dim, d_k, d_v, dropout=dropout)
    self.pos_ff = PositionwiseFF(dim, d_inner, dropout=dropout)

  def forward(self, enc_input, attn_mask=None):
    # attn_mask: (batch_size, )
    enc_output = self.attn(enc_input, enc_input, enc_input, attn_mask=attn_mask)
    enc_output = self.pos_ff(enc_output)
    return enc_output

class DecoderLayer(nn.Module):
  """Multi-head attention to both input_states and output_states."""

  def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
    super(DecoderLayer, self).__init__()
    self.slf_attn = MultiHeadAttn(n_head, d_model, d_k, d_v, dropout=dropout)
    self.enc_attn = MultiHeadAttn(n_head, d_model, d_k, d_v, dropout=dropout)
    self.pos_ffn = PositionwiseFF(d_model, d_inner, dropout=dropout)

  def forward(self, dec_input, enc_output, slf_attn_mask=None,
              dec_enc_attn_mask=None):
    dec_output = self.slf_attn(dec_input, dec_input, dec_input,
                               attn_mask=slf_attn_mask)
    dec_output = self.enc_attn(dec_output, enc_output, enc_output,
                               attn_mask=dec_enc_attn_mask)
    dec_output = self.pos_ffn(dec_output)
    return dec_output

