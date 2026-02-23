import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import os
from modules import *

class EmbedMatcher(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, 
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, 
                 aggregate='max', knn_k=10, knn_path=None):
        super(EmbedMatcher, self).__init__()

        self.actual_dim = embed.shape[1] if embed is not None else embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.actual_dim, padding_idx=num_symbols)
        self.knn_k = knn_k

        # Structural GCN
        self.gcn_w = nn.Linear(2 * self.actual_dim, self.actual_dim)
        self.gcn_b = nn.Parameter(torch.zeros(self.actual_dim))
        
        # Gating layer
        self.gate_layer = nn.Linear(2 * self.actual_dim, 1)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.gcn_w.weight)
        init.xavier_normal_(self.gate_layer.weight)

        if use_pretrain and embed is not None:
            logging.info(f'LOADING {embed.shape[1]}D KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False

        d_model = self.actual_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

        self.knn_neighbors = None 
        if knn_path and os.path.exists(knn_path):
            self.load_knn_index(knn_path)

    def load_knn_index(self, knn_path):
        try:
            self.knn_neighbors = torch.load(knn_path)
            logging.info(f'LOADED k-NN neighbors: {self.knn_neighbors.shape}')
        except Exception as e:
            logging.error(f"Failed to load k-NN index: {e}")

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
        relations = connections[:,:,0]
        entities = connections[:,:,1]

        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        # Distance-based pruning (Graph branch)
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

        # Semantic k-NN Branch
        if self.knn_neighbors is not None and entity_ids is not None:
            knn_idx = self.knn_neighbors[entity_ids].to(self.symbol_emb.weight.device)
            knn_ent_embeds = self.dropout(self.symbol_emb(knn_idx))
            
            # Prune semantic neighbors by distance to center
            center_embed = self.symbol_emb(entity_ids).unsqueeze(1)
            sim_sem = F.cosine_similarity(center_embed, knn_ent_embeds, dim=-1)
            topk_s = min(self.knn_k, sim_sem.size(1))
            _, topk_idx_s = torch.topk(sim_sem, k=topk_s, dim=-1)
            batch_idx_s = torch.arange(knn_idx.size(0), device=knn_idx.device).unsqueeze(1)
            knn_ent_embeds = knn_ent_embeds[batch_idx_s, topk_idx_s]

            knn_rel_embeds = self.dropout(self.symbol_emb(torch.zeros_like(knn_ent_embeds[:,:,0], dtype=torch.long)))
            concat_knn = torch.cat((knn_rel_embeds.unsqueeze(-1).expand_as(knn_ent_embeds), knn_ent_embeds), dim=-1)
            knn_mean = (self.gcn_w(concat_knn) + self.gcn_b).mean(dim=1).tanh()

            # Gating logic
            gate_input = torch.cat((structural_repr, knn_mean), dim=-1)
            alpha = torch.sigmoid(self.gate_layer(gate_input))
            return (1 - alpha) * structural_repr + alpha * knn_mean
        
        return structural_repr

    def forward(self, query, support, query_meta=None, support_meta=None):
        if query_meta is None:
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            return F.cosine_similarity(q_emb, s_emb.mean(0, keepdim=True).expand_as(q_emb))

        q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta[:4]
        s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta[:4]

        q_neighbor = torch.cat((self.neighbor_encoder(q_l_conn, q_l_deg, query[:,0]), 
                                self.neighbor_encoder(q_r_conn, q_r_deg, query[:,1])), dim=-1)
        s_neighbor = torch.cat((self.neighbor_encoder(s_l_conn, s_l_deg, support[:,0]), 
                                self.neighbor_encoder(s_r_conn, s_r_deg, support[:,1])), dim=-1)

        support_g = torch.mean(self.support_encoder(s_neighbor.unsqueeze(0)), dim=1)
        query_g = self.support_encoder(q_neighbor.unsqueeze(1)).squeeze(1)

        query_f = F.normalize(self.query_encoder(support_g.squeeze(0), query_g), p=2, dim=-1)
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)
        return torch.matmul(query_f, support_g.t()).squeeze(-1)
