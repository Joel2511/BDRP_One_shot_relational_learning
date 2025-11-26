import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        # --- Gating parameters ---
        self.gate_linear = nn.Linear(2*self.embed_dim, 1)
        init.xavier_normal_(self.gate_linear.weight)
        init.constant_(self.gate_linear.bias, 0)

        # --- Attention parameters ---
        self.attn_linear = nn.Linear(self.embed_dim, 1)
        init.xavier_normal_(self.attn_linear.weight)
        init.constant_(self.attn_linear.bias, 0)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

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
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)

        # --- Gating ---
        gate_values = torch.sigmoid(self.gate_linear(concat_embeds))
        gated_embeds = gate_values * concat_embeds

        # --- GCN projection ---
        projected = self.gcn_w(gated_embeds) + self.gcn_b

        # --- Attention ---
        attn_scores = self.attn_linear(projected).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)

        # Weighted sum of neighbors (remove division by num_neighbors)
        out = torch.sum(projected * attn_weights, dim=1)

        return out.tanh()

    def encode_neighbors(self, neighbors_1hop, neighbors_2hop, degrees):
        enc_1hop = self.neighbor_encoder(neighbors_1hop, degrees)
        enc_2hop = self.neighbor_encoder(neighbors_2hop, degrees)
        # Weighted combination: 1-hop slightly higher
        combined = enc_1hop * 1.2 + enc_2hop
        return combined

    def forward(self, query, support, query_meta=None, support_meta=None):
        if query_meta is None or support_meta is None:
            support = self.dropout(self.symbol_emb(support)).view(-1, 2*self.embed_dim)
            query = self.dropout(self.symbol_emb(query)).view(-1, 2*self.embed_dim)
            support = support.unsqueeze(0)
            support_g = self.support_encoder(support).squeeze(0)
            query_f = self.query_encoder(support_g, query)
            matching_scores = torch.matmul(query_f, support_g.t()).squeeze()
            return matching_scores

        query_left_1hop, query_left_2hop, query_left_degrees, query_right_1hop, query_right_2hop, query_right_degrees = query_meta
        support_left_1hop, support_left_2hop, support_left_degrees, support_right_1hop, support_right_2hop, support_right_degrees = support_meta

        query_left = self.encode_neighbors(query_left_1hop, query_left_2hop, query_left_degrees)
        query_right = self.encode_neighbors(query_right_1hop, query_right_2hop, query_right_degrees)

        support_left = self.encode_neighbors(support_left_1hop, support_left_2hop, support_left_degrees)
        support_right = self.encode_neighbors(support_right_1hop, support_right_2hop, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)

        support_g = self.support_encoder(support_neighbor)
        query_g = self.query_encoder(support_g, query_neighbor)

        support_g = torch.mean(support_g, dim=0, keepdim=True)

        matching_scores = torch.matmul(query_g, support_g.t()).squeeze()

        return matching_scores
