import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from modules import *

class EmbedMatcher(nn.Module):
    """
    Unified Gating-Based Matcher for Structural + Semantic Embeddings
    Handles multiple datasets (Medical, NELL, etc.) with dynamic embedding dims.
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, 
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, 
                 aggregate='max', knn_k=32, knn_path=None):
        super(EmbedMatcher, self).__init__()

        if embed is not None:
            self.actual_dim = embed.shape[1]
        else:
            self.actual_dim = embed_dim
        self.embed_dim = self.actual_dim 

        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.actual_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols
        self.knn_k = knn_k

        # GCN weights
        self.gcn_w = nn.Linear(2 * self.actual_dim, self.actual_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.actual_dim))
        # Sigmoid Gating Layer
        self.gate_layer = nn.Linear(2 * self.actual_dim, 1)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.gate_layer.weight)
        init.constant_(self.gate_layer.bias, 0)

        if use_pretrain and embed is not None:
            logging.info(f'LOADING {embed.shape[1]}D KB EMBEDDINGS INTO MATCHER')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False

        d_model = self.actual_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

        self.knn_neighbors = None 
        if knn_path is not None and os.path.exists(knn_path):
            self.knn_neighbors = torch.from_numpy(np.load(knn_path))
            logging.info(f'Loaded semantic k-NN neighbors: {self.knn_neighbors.shape}')

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
        num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)
        relations = connections[:,:,0]
        entities = connections[:,:,1]

        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        # Distance-based filtering for structural neighbors
        if entity_ids is not None:
            center_embed = self.symbol_emb(entity_ids).unsqueeze(1)
            sim = F.cosine_similarity(center_embed, ent_embeds, dim=-1)
            topk = min(self.knn_k, sim.size(1))
            _, topk_idx = torch.topk(sim, k=topk, dim=-1)
            batch_idx = torch.arange(entities.size(0), device=entities.device).unsqueeze(1)
            rel_embeds = rel_embeds[batch_idx, topk_idx]
            ent_embeds = ent_embeds[batch_idx, topk_idx]

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        structural_repr = (self.gcn_w(concat_embeds) + self.gcn_b).mean(dim=1).tanh()

        knn_mean = None
        if entity_ids is not None and self.knn_neighbors is not None:
            knn_idx = self.knn_neighbors[entity_ids].to(self.symbol_emb.weight.device)
            knn_ent_embeds = self.dropout(self.symbol_emb(knn_idx))
            knn_rel_embeds = self.dropout(self.symbol_emb(torch.zeros_like(knn_idx, dtype=torch.long)))
            concat_knn = torch.cat((knn_rel_embeds, knn_ent_embeds), dim=-1)
            knn_mean = (self.gcn_w(concat_knn) + self.gcn_b).mean(dim=1).tanh()

        # Gating logic
        if knn_mean is not None:
            gate_input = torch.cat((structural_repr, knn_mean), dim=-1)
            alpha = torch.sigmoid(self.gate_layer(gate_input))
            final = (1 - alpha) * structural_repr + alpha * knn_mean
        else:
            final = structural_repr

        return final

    def forward(self, query, support, query_meta=None, support_meta=None):
        if query_meta is None or support_meta is None:
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))

        q_h_ids, q_t_ids = query[:,0], query[:,1]
        s_h_ids, s_t_ids = support[:,0], support[:,1]

        q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
        s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta

        query_left = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids)
        query_right = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids)
        support_left = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)

        support_g = torch.mean(self.support_encoder(support_neighbor.unsqueeze(0)), dim=1)
        query_g = self.support_encoder(query_neighbor.unsqueeze(1)).squeeze(1)

        query_f = F.normalize(self.query_encoder(support_g.squeeze(0), query_g), p=2, dim=-1)
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)

        return torch.matmul(query_f, support_g.t()).squeeze(-1)
