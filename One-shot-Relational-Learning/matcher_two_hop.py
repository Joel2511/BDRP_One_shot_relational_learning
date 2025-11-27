import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings with stable 2‑hop + attention + gating
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1,
                                       embed_dim,
                                       padding_idx=self.pad_idx)
        self.aggregate = aggregate
        self.num_symbols = num_symbols

        # shared linear for neighbor projection
        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        # gating scalar per hop (learnable)
        self.hop_gate = nn.Parameter(torch.FloatTensor(2))  # gate for [1‑hop, 2‑hop]
        init.constant_(self.hop_gate, 1.0) # initialize both hops equal weight

        # attention score layer for neighbors
        self.attn_linear = nn.Linear(self.embed_dim, 1)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.attn_linear.weight)
        init.constant_(self.attn_linear.bias, 0)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        # connections: (batch, max_neighbors, 2)
        # num_neighbors: (batch,)
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]

        rel_emb = self.dropout(self.symbol_emb(relations))
        ent_emb = self.dropout(self.symbol_emb(entities))

        concat = torch.cat((rel_emb, ent_emb), dim=-1)  # (batch, neigh, 2*embed_dim)
        projected = self.gcn_w(concat) + self.gcn_b     # (batch, neigh, embed_dim)

        attn_scores = self.attn_linear(projected).squeeze(-1)  # (batch, neigh)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, neigh, 1)

        neigh_vec = torch.sum(projected * attn_weights, dim=1)  # (batch, embed_dim)
        return torch.tanh(neigh_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        if (query_meta is None) or (support_meta is None):
            # fallback: same as baseline
            support = self.dropout(self.symbol_emb(support)).view(-1, 2*self.embed_dim)
            query = self.dropout(self.symbol_emb(query)).view(-1, 2*self.embed_dim)
            support = support.unsqueeze(0)
            support_g = self.support_encoder(support).squeeze(0)
            query_f = self.query_encoder(support_g, query)
            return torch.matmul(query_f, support_g.t()).squeeze()

        (q_l1, q_l2, q_deg_l, q_r1, q_r2, q_deg_r) = query_meta
        (s_l1, s_l2, s_deg_l, s_r1, s_r2, s_deg_r) = support_meta

        # encode neighbors per hop
        q_left_1 = self.neighbor_encoder(q_l1, q_deg_l)
        q_left_2 = self.neighbor_encoder(q_l2, q_deg_l)
        q_right_1 = self.neighbor_encoder(q_r1, q_deg_r)
        q_right_2 = self.neighbor_encoder(q_r2, q_deg_r)

        s_left_1 = self.neighbor_encoder(s_l1, s_deg_l)
        s_left_2 = self.neighbor_encoder(s_l2, s_deg_l)
        s_right_1 = self.neighbor_encoder(s_r1, s_deg_r)
        s_right_2 = self.neighbor_encoder(s_r2, s_deg_r)

        # combine hops using gating
        # gate weights softmax normalized hop-wise
        gate_weights = F.softmax(self.hop_gate, dim=0)  # two weights sum to 1
        q_left = gate_weights[0] * q_left_1 + gate_weights[1] * q_left_2
        q_right = gate_weights[0] * q_right_1 + gate_weights[1] * q_right_2
        s_left = gate_weights[0] * s_left_1 + gate_weights[1] * s_left_2
        s_right = gate_weights[0] * s_right_1 + gate_weights[1] * s_right_2

        query_neighbor = torch.cat((q_left, q_right), dim=-1)
        support_neighbor = torch.cat((s_left, s_right), dim=-1)

        support_g = self.support_encoder(support_neighbor)
        query_g = self.query_encoder(support_g, query_neighbor)
        support_g = torch.mean(support_g, dim=0, keepdim=True)

        matching_scores = torch.matmul(query_g, support_g.t()).squeeze()
        return matching_scores
