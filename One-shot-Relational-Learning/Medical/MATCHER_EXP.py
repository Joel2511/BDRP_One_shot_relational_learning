import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from modules import *

class EmbedMatcher(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True,
                 embed=None, dropout=0.2, batch_size=64,
                 finetune=False, semantic_matrix=None):

        super(EmbedMatcher, self).__init__()
    
        self.actual_dim = embed_dim
        self.dropout_val = dropout  # Store the float value separately
        self.batch_size = batch_size
        self.finetune = finetune
        self.semantic_matrix = semantic_matrix
        
        # --- RESEARCH-SAFE INITIALIZATION ---
        # 1. Determine num_symbols from the actual incoming embed shape
        # This handles the 16902 vs 16904 mismatch automatically for Medical.
        self.num_symbols = embed.shape[0] if embed is not None else num_symbols
        self.pad_idx = self.num_symbols - 1
    
        self.symbol_emb = nn.Embedding(self.num_symbols, embed_dim, padding_idx=0)
    
        if use_pretrain and embed is not None:
            logging.info(f'LOADING KB EMBEDDINGS: {embed.shape[0]}x{embed.shape[1]}')
            # Copy pre-trained weights into the (potentially resized) table
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False
    
        # The actual dropout module module
        self.dropout_layer = nn.Dropout(dropout)
    
        # Structural GCN components used in neighbor_encoder
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.zeros(embed_dim))
        init.xavier_normal_(self.gcn_w.weight)

        # Learnable semantic projection (Semantic Anchor logic)
        self.semantic_proj = None
        if semantic_matrix is not None:
            semantic_dim = semantic_matrix.shape[1]
            self.semantic_proj = nn.Linear(semantic_dim, embed_dim, bias=False)
            nn.init.xavier_normal_(self.semantic_proj.weight)

        # Encoders from modules.py (LSTM Matching Logic)
        d_model = 2 * embed_dim
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, 2)

        # Internal state for k-NN Distance Filter
        self.knn_neighbors = None
        self.knn_k = 32

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
        num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)
        relations = connections[:, :, 0]
        entities  = connections[:, :, 1]

        # FIX: Correctly call the dropout layer module
        rel_embeds = self.dropout_layer(self.symbol_emb(relations))
        ent_embeds = self.dropout_layer(self.symbol_emb(entities))

        # --- DISTANCE FILTER ---
        if entity_ids is not None:
            center_embed = self.symbol_emb(entity_ids).unsqueeze(1)
            sim = F.cosine_similarity(center_embed, ent_embeds, dim=-1)
            # Prune bottom tier nodes based on geometric proximity
            topk = min(self.knn_k, sim.size(1))
            _, topk_idx = torch.topk(sim, k=topk, dim=-1)
            
            batch_idx = torch.arange(entities.size(0), device=entities.device).unsqueeze(1)
            rel_embeds = rel_embeds[batch_idx, topk_idx]
            ent_embeds = ent_embeds[batch_idx, topk_idx]

        # Aggregation of filtered neighbors
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        structural_repr = (self.gcn_w(concat_embeds) + self.gcn_b).mean(dim=1).tanh()

        return structural_repr

    def forward(self, query, support, query_meta=None, support_meta=None):
        # --- SEMANTIC ADDITION (Inductive Lifeline) ---
        if self.semantic_proj is not None and self.semantic_matrix is not None:
            device = self.symbol_emb.weight.device
            sem_tensor = torch.from_numpy(self.semantic_matrix).to(device).float()
            projected = self.semantic_proj(sem_tensor)
    
            # Add semantic anchors to entity embeddings (Skip Relation block at index 0)
            rel_count = projected.shape[0]
            max_idx = min(rel_count + 1, self.symbol_emb.weight.shape[0])
            self.symbol_emb.weight.data[1:max_idx] += projected[:max_idx-1]
    
        if query_meta is None or support_meta is None:
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))
    
        # Tuple handling for cross-dataset meta-data compatibility
        if len(query_meta) == 6:
            q_l_conn, _, q_l_deg, q_r_conn, _, q_r_deg = query_meta
            s_l_conn, _, s_l_deg, s_r_conn, _, s_r_deg = support_meta
        else:
            q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
            s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta
    
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]
    
        # Stage 1: Filtered Context Extraction
        query_left    = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids)
        query_right   = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids)
        support_left  = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids)
    
        # Stage 2: Concatenation and LSTM Refinement
        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)
    
        # Process through G-Matching Matcher (LSTM)
        support_g = torch.mean(self.support_encoder(support_neighbor.unsqueeze(0)), dim=1)
        query_g = self.support_encoder(query_neighbor.unsqueeze(1)).squeeze(1)
    
        # Stage 3: Induction and Similarity Scoring
        query_f = F.normalize(self.query_encoder(support_g.squeeze(0), query_g), p=2, dim=-1)
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)
    
        return torch.matmul(query_f, support_g.t()).squeeze(-1)
