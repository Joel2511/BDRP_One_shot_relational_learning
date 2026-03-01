import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from modules import *


class EmbedMatcher(nn.Module):
    """
    Unified Gating-Based Matcher for Structural + Semantic Embeddings.
    Handles NELL-One, FB15k-237, Medical/OntoOmics, ATOMIC.

    Gate only fires when knn_neighbors is loaded (knn_path provided).
    Otherwise pure structural GCN — same behaviour as both originals.
    """

    def __init__(self, embed_dim, num_symbols, use_pretrain=True,
             embed=None, dropout=0.2, batch_size=64,
             finetune=False, semantic_matrix=None):

        super(EmbedMatcher, self).__init__()
    
        self.actual_dim = embed_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.finetune = finetune
        self.semantic_matrix = semantic_matrix
    
        # --- Safe defaults for kNN ---
        self.knn_neighbors = None
        self.knn_k = 0
        self.pad_idx = 0  # if pad tensor is used in neighbor_encoder
    
        self.symbol_emb = nn.Embedding(embed.shape[0], embed_dim, padding_idx=0)
    
        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False
    
        self.dropout_layer = nn.Dropout(dropout)
    
        # Learnable semantic projection (NEW)
        self.semantic_proj = None
        if semantic_matrix is not None:
            semantic_dim = semantic_matrix.shape[1]
            self.semantic_proj = nn.Linear(semantic_dim, embed_dim, bias=False)
            nn.init.xavier_normal_(self.semantic_proj.weight)
    
    
    def _load_knn(self, knn_path):
        try:
            if knn_path.endswith('.pt') or knn_path.endswith('.pth'):
                self.knn_neighbors = torch.load(knn_path)
            else:
                self.knn_neighbors = torch.from_numpy(np.load(knn_path))
            self.knn_k = self.knn_neighbors.shape[1]
            logging.info(f'Loaded k-NN neighbors: {self.knn_neighbors.shape}')
        except Exception as e:
            logging.error(f'Failed to load k-NN index from {knn_path}: {e}')
            # fallback to safe defaults
            self.knn_neighbors = None
            self.knn_k = 0

    # ------------------------------------------------------------------
    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
        num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)
        relations = connections[:, :, 0]
        entities  = connections[:, :, 1]

        # FIXED: use dropout_layer instead of float
        rel_embeds = self.dropout_layer(self.symbol_emb(relations))
        ent_embeds = self.dropout_layer(self.symbol_emb(entities))

        # Cosine-similarity-based topk neighbor filtering
        if entity_ids is not None:
            center_embed = self.symbol_emb(entity_ids).unsqueeze(1)
            sim  = F.cosine_similarity(center_embed, ent_embeds, dim=-1)
            topk = min(self.knn_k, sim.size(1))
            _, topk_idx = torch.topk(sim, k=topk, dim=-1)
            batch_idx  = torch.arange(entities.size(0), device=entities.device).unsqueeze(1)
            rel_embeds = rel_embeds[batch_idx, topk_idx]
            ent_embeds = ent_embeds[batch_idx, topk_idx]

        concat_embeds  = torch.cat((rel_embeds, ent_embeds), dim=-1)
        structural_repr = (self.gcn_w(concat_embeds) + self.gcn_b).mean(dim=1).tanh()

        # Gate path — only active when knn_neighbors loaded
        knn_mean = None
        if entity_ids is not None and self.knn_neighbors is not None:
            knn_idx = self.knn_neighbors[entity_ids].to(self.symbol_emb.weight.device)

            # FIXED: use dropout_layer
            knn_ent_embeds = self.dropout_layer(self.symbol_emb(knn_idx))

            # Topk within knn neighbours
            center_embed = self.symbol_emb(entity_ids).unsqueeze(1)
            sim  = F.cosine_similarity(center_embed, knn_ent_embeds, dim=-1)
            topk = min(self.knn_k, sim.size(1))
            _, topk_idx = torch.topk(sim, k=topk, dim=-1)
            batch_idx      = torch.arange(knn_idx.size(0), device=knn_idx.device).unsqueeze(1)
            knn_ent_embeds = knn_ent_embeds[batch_idx, topk_idx]

            # Use pad embedding as pseudo-relation for knn branch
            pad_tensor    = torch.full_like(knn_idx, self.pad_idx, dtype=torch.long,
                                            device=self.symbol_emb.weight.device)

            # FIXED: use dropout_layer
            knn_rel_embeds = self.dropout_layer(self.symbol_emb(pad_tensor))

            concat_knn = torch.cat((knn_rel_embeds, knn_ent_embeds), dim=-1)
            knn_mean   = (self.gcn_w(concat_knn) + self.gcn_b).mean(dim=1).tanh()

        if knn_mean is not None:
            gate_input = torch.cat((structural_repr, knn_mean), dim=-1)
            alpha      = torch.sigmoid(self.gate_layer(gate_input))
            return (1 - alpha) * structural_repr + alpha * knn_mean
        else:
            return structural_repr

    # ------------------------------------------------------------------
    def forward(self, query, support, query_meta=None, support_meta=None):
        # --- Inject semantic projection into embedding table ---
        if self.semantic_proj is not None and self.semantic_matrix is not None:
            device = self.symbol_emb.weight.device
            sem_tensor = torch.from_numpy(self.semantic_matrix).float().to(device)
            projected = self.semantic_proj(sem_tensor)
    
            # Add projected semantic to entity portion only
            rel_count = projected.shape[0]
            self.symbol_emb.weight.data[1:rel_count+1] += projected
    
        if query_meta is None or support_meta is None:
            q_emb  = self.symbol_emb(query).view(query.size(0), -1)
            s_emb  = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))
    
        if len(query_meta) == 6:
            q_l_conn, _, q_l_deg, q_r_conn, _, q_r_deg = query_meta
            s_l_conn, _, s_l_deg, s_r_conn, _, s_r_deg = support_meta
        else:
            q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
            s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta
    
        q_h_ids = query[:, 0]
        q_t_ids = query[:, 1]
        s_h_ids = support[:, 0]
        s_t_ids = support[:, 1]
    
        query_left    = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids)
        query_right   = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids)
        support_left  = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids)
    
        query_neighbor   = torch.cat((query_left,   query_right),   dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)
    
        support_g = torch.mean(self.support_encoder(support_neighbor.unsqueeze(0)), dim=1)
        query_g   = self.support_encoder(query_neighbor.unsqueeze(1)).squeeze(1)
    
        query_f   = F.normalize(self.query_encoder(support_g.squeeze(0), query_g), p=2, dim=-1)
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)
    
        return torch.matmul(query_f, support_g.t()).squeeze(-1)
