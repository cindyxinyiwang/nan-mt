import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch

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
		self.softmax = nn.Softmax()

	def forward(self, q, k, v, attn_mask=None):
		attn = torch.bmm(q, k.transpose(1, 2)) / self.temp # (n_head*batch_size, len_q, len_k)
		
		size = attn.size()
		assert len(size) > 2
		# doing softmax along dim 1
		attn = self.softmax(attn.view(size[0]*size[1], -1)).view(size[0], size[1], -1)

		attn = self.dropout(attn)
		output = torch.bmm(attn, v) # (n_head*batch_size, len_q, dim_v); len_k == len_v

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

		init.xavier_normal(self.w_q)
		init.xavier_normal(self.w_k)
		init.xavier_normal(self.w_v)
		init.xavier_normal(self.proj.weight)

	def forward(self, q, k, v, attn_mask=None):

		batch_size, len_q, dim = q.size()
		batch_size, len_k, dim = k.size()
		batch_size, len_v, dim = v.size()

		q_batch = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, dim) # (n_head * (batch_size * len_q) * dim)
		k_batch = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, dim)
		v_batch = v.repeat(self.n_head, 1, 1).view(self.n_head, -1, dim)

		q_batch = torch.bmm(q_batch, self.w_q).view(-1, len_q, self.d_k) # (n_head * (batch_size * len_q) * d_k) => (n_head*batch_size * len_q * d_k)
		k_batch = torch.bmm(k_batch, self.w_k).view(-1, len_k, self.d_k)
		v_batch = torch.bmm(v_batch, self.w_v).view(-1, len_v, self.d_v)

		if attn_mask:
			attn_mask = attn_mask.repeat(n_head, 1, 1)
		outputs = self.attention(q_batch, k_batch, v_batch, attn_mask=attn_mask) #(n_head*batch_size, len_q, dim_v)

		outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1) # (n_heads, batch_size, len_q, dim_v)=>(batch_size, len_q, n_head*dim_v)

		outputs = outputs.view(batch_size*len_q, -1)
		outputs = self.proj(outputs).view(batch_size, len_q, -1) # (batch_size, len_q, dim)
		outputs = self.dropout(outputs)

		return outputs

class PositionwiseFF(nn.Module):
	def __init__(self, d_hid, d_inner, dropout=0.1):
		super(PositionwiseFF, self).__init__()
		self.w_1 = nn.Conv1d(d_hid, d_inner, 1)
		self.w_2 = nn.Conv1d(d_inner, d_hid, 1)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()

	def forward(self, x):
		output = self.relu(self.w_1(x.transpose(1, 2)))
		output = self.w_2(output).transpose(2, 1)
		output = self.dropout(output)
		return output

class EncoderLayer(nn.Module):
	'''
		Compose multi-head attention and positionwise feeding
	'''
	def __init__(self, dim, d_inner, n_head, d_k, d_v, dropout=0.1):
		super(EncoderLayer, self).__init__()
		self.attn = MultiHeadAttn(n_head, dim, d_k, d_v, dropout=dropout)
		self.pos_ff = PositionwiseFF(dim, d_inner, dropout=dropout)

	def forward(self, enc_input, attn_mask=None):
		enc_output = self.attn(enc_input, enc_input, enc_input, attn_mask=attn_mask)
		enc_output = self.pos_ff(enc_output)
		return enc_output

