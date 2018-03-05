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


INF = np.float32(np.inf)
NEG_INF = float(np.float32(-np.inf))


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
    pos_emb_sin = torch.sin(pos_emb_indices / self.freq).view(
      batch_size, max_len, -1, 1)
    pos_emb_cos = torch.cos(pos_emb_indices / self.freq).view(
      batch_size, max_len, -1, 1)
    pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=3).view(
      batch_size, max_len, self.hparams.d_word_vec)
    pos_emb.data.masked_fill_(mask, float(0))

    return pos_emb


class LayerNormalization(nn.Module):
  def __init__(self, d_hid, eps=1e-5):
    super(LayerNormalization, self).__init__()

    self.eps = eps
    self.scale = nn.Parameter(torch.ones(d_hid), requires_grad=True)
    self.offset= nn.Parameter(torch.zeros(d_hid), requires_grad=True)

  def forward(self, x):
    assert x.dim() >= 2
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.scale * (x - mean) / (std + self.eps) + self.offset


class ScaledDotProdAttn(nn.Module):
  def __init__(self, dim, dropout=0.1):
    super(ScaledDotProdAttn, self).__init__()
    self.temp = np.power(dim, 0.5)
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, q, k, v, attn_mask=None):
    """Compute Softmax(q * k.T / sqrt(dim)) * v

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].
    
    Note: batch_size may be n_heads * batch_size, but we don't care.
    
    Must have:
      d_q == d_k
      len_k == len_v

    Returns:
      attn: [batch_size, d_v].
    """

    batch_q, len_q, d_q = q.size()
    batch_k, len_k, d_k = k.size()
    batch_v, len_v, d_v = v.size()

    assert batch_q == batch_k and batch_q == batch_v
    assert d_q == d_k and len_k == len_v

    # [batch_size, len_q, len_k]
    attn = torch.bmm(q, k.transpose(1, 2)) / self.temp
    first_attn = torch.mm(q[0], k[0].transpose(0, 1))

    # attn_mask: [batch_size, len_q, len_k]
    if not attn_mask is None:
      attn.data.masked_fill_(attn_mask, -float("inf"))
    size = attn.size()
    assert len(size) > 2 and len_q == size[1] and len_k == size[2]

    # softmax along the len_k dimension
    # [batch_size, len_q, len_k]
    attn = self.softmax(attn.view(size[0] * size[1], -1)).view(
      batch_q, len_q, len_k)

    # [batch_size, len_q, len_k == len_v]
    attn = self.dropout(attn)

    # [batch_size, len_q, d_v]
    output = torch.bmm(attn, v)

    return output


class MultiHeadAttn(nn.Module):
  def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
    super(MultiHeadAttn, self).__init__()

    self.n_heads = n_heads
    self.d_q = d_k
    self.d_k = d_k
    self.d_v = d_v
    self.d_model = d_model

    self.w_q = nn.Parameter(torch.FloatTensor(n_heads, self.d_model, self.d_q))
    self.w_k = nn.Parameter(torch.FloatTensor(n_heads, self.d_model, self.d_k))
    self.w_v = nn.Parameter(torch.FloatTensor(n_heads, self.d_model, self.d_v))

    self.attention = ScaledDotProdAttn(self.d_model, dropout=dropout)

    # projection of concatenated attn
    self.w_proj = nn.Linear(n_heads * self.d_v, self.d_model, bias=True)

    self.layer_norm = LayerNormalization(self.d_model)

    init.xavier_normal(self.w_q)
    init.xavier_normal(self.w_k)
    init.xavier_normal(self.w_v)
    init.xavier_normal(self.w_proj.weight)

  def forward(self, q, k, v, attn_mask=None):
    """Performs the following computations:

         head[i] = Attention(q * w_q[i], k * w_k[i], v * w_v[i])
         outputs = concat(all head[i]) * self.w_proj

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].

    Must have: len_k == len_v
    Note: This batch_size is in general NOT the training batch_size, as
      both sentences and time steps are batched together for efficiency.

    Returns:
      outputs: [batch_size, len_q, d_v].
    """

    residual = q 
    
    batch_size, len_q, d_q = q.size()
    batch_size, len_k, d_k = k.size()
    batch_size, len_v, d_v = v.size()

    assert (d_q == self.d_model and
            d_k == self.d_model and
            d_v == self.d_model and
            len_k == len_v)

    # [n_heads, batch_size * len_{q,k,v}, d_model]
    q_batch = q.repeat(self.n_heads, 1, 1).view(self.n_heads, batch_size * len_q, self.d_model)
    k_batch = k.repeat(self.n_heads, 1, 1).view(self.n_heads, batch_size * len_k, self.d_model)
    v_batch = v.repeat(self.n_heads, 1, 1).view(self.n_heads, batch_size * len_v, self.d_model)

    # [n_heads, batch_size * len_, d_] -> [n_heads * batch_size, len_, d_]
    q_batch = torch.bmm(q_batch, self.w_q).view(self.n_heads * batch_size, len_q, self.d_q)
    k_batch = torch.bmm(k_batch, self.w_k).view(self.n_heads * batch_size, len_k, self.d_k)
    v_batch = torch.bmm(v_batch, self.w_v).view(self.n_heads * batch_size, len_v, self.d_v)

    # [n_heads * batch_size, len_q, len_k]
    if not attn_mask is None:
      attn_mask = attn_mask.repeat(self.n_heads, 1, 1)
    outputs = self.attention(q_batch, k_batch, v_batch, attn_mask=attn_mask)

    # [n_heads * batch_size, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
    outputs = outputs.view(self.n_heads, batch_size, len_q, self.d_v).permute(
      1, 2, 0, 3).contiguous().view(batch_size, len_q, self.n_heads * self.d_v)

    # [batch_size, len_q, d_model]
    outputs = outputs.view(batch_size * len_q, self.n_heads * self.d_v)
    outputs = self.w_proj(outputs).view(batch_size, len_q, self.d_model)

    # residual
    # outputs = self.layer_norm(outputs + residual)
    outputs = outputs + residual

    return outputs


class PositionwiseFF(nn.Module):
  def __init__(self, d_model, d_inner, dropout=0.1):
    super(PositionwiseFF, self).__init__()
    self.d_model = d_model
    self.d_inner = d_inner
    self.w_1 = nn.Linear(d_model, d_inner, bias=False)
    self.w_2 = nn.Linear(d_inner, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()
    self.layer_norm = LayerNormalization(d_model)

    init.xavier_normal(self.w_1.weight)
    init.xavier_normal(self.w_2.weight)

  def forward(self, x):
    residual = x
    batch_size, x_len, d_model = x.size()
    x = self.relu(self.w_1(x.view(-1, d_model)))
    x = self.w_2(x).view(batch_size, x_len, d_model)
    x = self.dropout(x)
    x += residual
    x = self.layer_norm(x)
    return x


class EncoderLayer(nn.Module):
  """Compose multi-head attention and positionwise feeding."""

  def __init__(self, d_model, d_inner, n_heads, d_k, d_v, dropout=0.1):
    super(EncoderLayer, self).__init__()
    self.attn = MultiHeadAttn(n_heads, d_model, d_k, d_v, dropout=dropout)
    self.pos_ff = PositionwiseFF(d_model, d_inner, dropout=dropout)

  def forward(self, enc_input, attn_mask=None):
    """Normal forward pass.

    Args:
      enc_input: [batch_size, x_len, d_model].
      attn_mask: [batch_size, x_len, x_len].
    """

    enc_output = self.attn(enc_input, enc_input, enc_input, attn_mask=attn_mask)
    enc_output = self.pos_ff(enc_output)
    return enc_output


class DecoderLayer(nn.Module):
  """Multi-head attention to both input_states and output_states."""

  def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
    super(DecoderLayer, self).__init__()
    self.y_attn = MultiHeadAttn(n_head, d_model, d_k, d_v, dropout=dropout)
    self.x_attn = MultiHeadAttn(n_head, d_model, d_k, d_v, dropout=dropout)
    self.pos_ffn = PositionwiseFF(d_model, d_inner, dropout=dropout)

  def forward(self, dec_input, enc_output, y_attn_mask=None, x_attn_mask=None):
    """Decoder.

    Args:
      y_attn_mask: self attention mask.
      x_attn_mask: decoder-encoder attention mask.
    """

    output = self.y_attn(dec_input, dec_input, dec_input, attn_mask=y_attn_mask)
    output = self.x_attn(output, enc_output, enc_output, attn_mask=x_attn_mask)
    output = self.pos_ffn(output)
    return output

