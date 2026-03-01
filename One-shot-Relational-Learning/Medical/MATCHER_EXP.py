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
    
    The Gate only fires when knn_neighbors is loaded. 
    Otherwise, it defaults to pure structural GCN.
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, 
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, 
                 knn_path=None, semantic_matrix=None):
        super(EmbedMatcher, self).__init__()
        
        self.actual_dim = embed_dim
        self.dropout_rate = dropout
        self.batch_size = batch_size
        self.finetune = finetune
        self.num_symbols = num_symbols
        self.pad_idx = 0
        
        # 1. Embedding Layer
        self.symbol_emb = nn.Embedding(num_symbols, embed_dim, padding_idx=self.pad_idx)
        if use_pretrain and embed is not None:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)

        # 2. GCN & Gating Layers
        # These must be defined to be used in neighbor_encoder
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.zeros(embed_dim))
        self.gate_layer = nn.Linear(2 * embed_dim, 1)
        
        # 3. Semantic Projection
        self.semantic_matrix = semantic_matrix
        self.semantic_proj = None
        if semantic_matrix is not None:
            semantic_dim = semantic_matrix.shape[1]
            self.semantic_proj = nn.Linear(semantic_dim, embed_dim, bias=False)
            init.xavier_normal_(self.semantic_proj.weight)

        # 4. Encoders (Assumed from modules.py)
        self.support_encoder = SupportEncoder(embed_dim * 2, embed_dim * 4, dropout)
        self.query_encoder = QueryEncoder(embed_dim * 2, process_steps)

        # 5. k-NN Setup
        self.knn_neighbors = None
        self.knn_k = 0
        if knn_path:
            self._load_knn(knn_path)

        # Initialize weights
        init.xavier_normal_(self.gcn_w.weight)
        init.xavier_normal_(self.gate_layer.weight)

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

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
        # connections shape: (batch, max_neighbors, 2)
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]

        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        # Filter neighbors based on similarity to center entity
        if entity_ids is not None:
            center_embed = self.symbol_emb(entity_ids).unsqueeze(1)
            sim = F.cosine_similarity(center_embed, ent_embeds, dim=-1)
            
            # Dynamically determine top-k
            current_k = min(self.knn_k if self.knn_k > 0 else 32, sim.size(1))
            _, topk_idx = torch.topk(sim, k=current_k, dim=-1)
            
            batch_idx = torch.arange(entities.size(0), device=entities.device).unsqueeze(1)
            rel_embeds = rel_embeds[batch_idx, topk_idx]
            ent_embeds = ent_embeds[batch_idx, topk_idx]

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        # Apply GCN: tanh(W*x + b)
        structural_repr = (self.gcn_w(concat_embeds) + self.gcn_b).mean(dim=1).tanh()

        # --- Gating Logic (Semantic/k-NN path) ---
        knn_mean = None
        if entity_ids is not None and self.knn_neighbors is not None:
            knn_idx = self.knn_neighbors[entity_ids].to(entities.device)
            knn_ent_embeds = self.dropout(self.symbol_emb(knn_idx))

            # Similarity filtering for k-NN branch
            center_embed = self.symbol_emb(entity_ids).unsqueeze(1)
            sim_knn = F.cosine_similarity(center_embed, knn_ent_embeds, dim=-1)
            knn_top_k = min(self.knn_k, sim_knn.size(1))
            _, knn_topk_idx = torch.topk(sim_knn, k=knn_top_k, dim=-1)
            
            batch_idx_knn = torch.arange(knn_idx.size(0), device=knn_idx.device).unsqueeze(1)
            knn_ent_embeds = knn_ent_embeds[batch_idx_knn, knn_topk_idx]

            # Use padding index as pseudo-relation for k-NN branch
            pad_tensor = torch.full_like(knn_idx, self.pad_idx, dtype=torch.long, device=entities.device)
            knn_rel_embeds = self.dropout(self.symbol_emb(pad_tensor))
            
            concat_knn = torch.cat((knn_rel_embeds, knn_ent_embeds), dim=-1)
            knn_mean = (self.gcn_w(concat_knn) + self.gcn_b).mean(dim=1).tanh()

        if knn_mean is not None:
            gate_input = torch.cat((structural_repr, knn_mean), dim=-1)
            alpha = torch.sigmoid(self.gate_layer(gate_input))
            return (1 - alpha) * structural_repr + alpha * knn_mean
        
        return structural_repr

    def forward(self, query, support, query_meta=None, support_meta=None):
        # 1. Update embedding table with semantic projections if available
        if self.semantic_proj is not None and self.semantic_matrix is not None:
            device = self.symbol_emb.weight.device
            sem_tensor = torch.from_numpy(self.semantic_matrix).to(device).float()
            projected = self.semantic_proj(sem_tensor)
            
            # In-place update (Careful: might affect gradients if not handled properly)
            rel_count = projected.shape[0]
            self.symbol_emb.weight.data[1:rel_count+1] += projected

        # 2. Simple embedding lookup if no metadata provided
        if query_meta is None or support_meta is None:
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))

        # 3. Unpack Meta-Information (supports 4-tuple or 6-tuple)
        if len(query_meta) == 6:
            q_l_conn, _, q_l_deg, q_r_conn, _, q_r_deg = query_meta
            s_l_conn, _, s_l_deg, s_r_conn, _, s_r_deg = support_meta
        else:
            q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
            s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta

        # Head/Tail IDs
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        # Encode Neighbors
        query_left = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids)
        query_right = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids)
        support_left = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids)

        # Final representation construction
        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)

        # Support representation (Mean pooling across support set)
        support_g = self.support_encoder(support_neighbor.unsqueeze(0))
        support_g = torch.mean(support_g, dim=1)

        # Query representation
        query_g = self.support_encoder(query_neighbor.unsqueeze(1)).squeeze(1)
        
        # Matching
        query_f = F.normalize(self.query_encoder(support_g.squeeze(0), query_g), p=2, dim=-1)
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)

        return torch.matmul(query_f, support_g.t()).squeeze(-1)
