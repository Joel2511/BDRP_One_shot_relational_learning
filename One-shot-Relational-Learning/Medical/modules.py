import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import optim
from torch.autograd import Variable

class Path(nn.Module):
    """Convolution to encode every path between an entity pair (supports 1-hop and 2-hop)"""
    def __init__(self, input_dim, num_symbols, use_pretrain=True, embed_path='', dropout=0.5, k_sizes=[3], k_num=100):
        super(Path, self).__init__()
        self.symbol_emb = nn.Embedding(num_symbols + 1, input_dim, padding_idx=num_symbols)
        self.k_sizes = k_sizes
        self.k_num = k_num

        if use_pretrain and embed_path:
            emb_np = np.loadtxt(embed_path)
            self.symbol_emb.weight.data.copy_(torch.from_numpy(emb_np))
            self.symbol_emb.weight.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1, self.k_num, (k, input_dim)) for k in self.k_sizes])
        self.dropout = nn.Dropout(dropout)

    def forward(self, path):
        """
        path: batch * max_len
        supports 1-hop or 2-hop input
        """
        path = self.symbol_emb(path)
        path = path.unsqueeze(1)  # (B, 1, W, D)
        convs = [F.relu(conv(path)).squeeze(3) for conv in self.convs]  # (B, k_num, W-(k-1))
        pools = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in convs]  # (B, k_num)
        path = torch.cat(pools, 1)
        path = self.dropout(path)
        return path


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1):
        super().__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid))
        self.b_2 = nn.Parameter(torch.zeros(d_hid))

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu) / (sigma + self.eps)
        ln_out = ln_out * self.a_2 + self.b_2
        return ln_out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = nn.Linear(n_head * d_v, d_model)
        init.xavier_normal_(self.proj.weight)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q
        mb_size, len_q, d_model = q.size()
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, k.size(1), d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, v.size(1), d_v)

        if attn_mask is not None:
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask.repeat(n_head, 1, 1))
        else:
            outputs, attns = self.attention(q_s, k_s, v_s)

        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        return self.layer_norm(outputs + residual), attns


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class SupportEncoder(nn.Module):
    """Encodes support set embeddings (context-aware)"""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)
        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        return enc_output, enc_slf_attn


class ContextAwareEncoder(nn.Module):
    """Self-attention encoder for 1-hop and 2-hop neighbors"""
    def __init__(self, num_layers, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, elements, enc_slf_attn_mask=None):
        enc_output = elements
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
        return enc_output


class QueryEncoder(nn.Module):
    def __init__(self, input_dim, process_step=4):
        super().__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(self, support, query):
            if self.process_step == 0:
                return query
            batch_size = query.size(0)
            
            # Ensure support is 2D [1, input_dim] even in one-shot
            if support.dim() == 1:
                support = support.unsqueeze(0)
                
            h_r = torch.zeros(batch_size, 2 * self.input_dim).to(query.device)
            c = torch.zeros(batch_size, 2 * self.input_dim).to(query.device)
            
            for _ in range(self.process_step):
                h_r_, c = self.process(query, (h_r, c))
                h = query + h_r_[:, :self.input_dim]
                
                # --- DIMENSION FIX ---
                # Matmul [B, D] x [D, 1] -> [B, 1]
                scores = torch.matmul(h, support.t())
                attn = F.softmax(scores, dim=1) 
                
                # r: [B, D]
                r = torch.matmul(attn, support)
                h_r = torch.cat((h, r), dim=1)
            return h


if __name__ == '__main__':
    # test code for 2-hop
    encoder = ContextAwareEncoder(2, 100, 200, 4, 25, 25)
    support = Variable(torch.randn(128, 200, 100))
    print(encoder(support).size())
