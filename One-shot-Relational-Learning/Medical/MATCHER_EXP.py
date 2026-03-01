import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from modules import *

class EmbedMatcher(nn.Module):
    """
    Unified Gating-Based Matcher for Structural + Semantic Embeddings.
    Compatible with Medical, NELL, FB15k, ATOMIC, etc.
    """

    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False,
                 aggregate='max', knn_k=32, knn_path=None, semantic_matrix=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_symbols = num_symbols
        self.pad_idx = num_symbols - 1
        self.batch_size = batch_size
        self.aggregate = aggregate
        self.knn_k = knn_k

        # Embeddings
        self.symbol_emb = nn.Embedding(num_symbols, embed_dim, padding_idx=self.pad_idx)
        if use_pretrain and embed is not None:
            logging.info(f'LOADING {embed.shape[0]}x{embed.shape[1]} KB EMBEDDINGS')
        
            # --- DYNAMIC RESIZE START ---
            if self.symbol_emb.weight.shape[0] != embed.shape[0]:
                logging.info(f"RESIZING: {self.symbol_emb.weight.shape[0]} -> {embed.shape[0]}")
                self.num_symbols = embed.shape[0]
                # Update pad_idx to the new last index
                self.pad_idx = self.num_symbols - 1 
                self.symbol_emb = nn.Embedding(self.num_symbols, self.embed_dim, padding_idx=self.pad_idx)
            # --- DYNAMIC RESIZE END ---
        
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False

        # Structural GCN & Gate
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.zeros(embed_dim))
        self.gate_layer = nn.Linear(2 * embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.gate_layer.weight)
        init.constant_(self.gate_layer.bias, 0)

        # Query/Support encoders
        d_model = 2 * embed_dim
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

        # kNN
        self.knn_neighbors = None
        if knn_path is not None:
            self.load_knn_index(knn_path)

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
        num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)

        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        if entity_ids is not None:
            # SAFE: Clamp entity IDs to valid range
            entity_ids_safe = entity_ids.clone()
            entity_ids_safe[entity_ids_safe < 0] = self.pad_idx
            entity_ids_safe[entity_ids_safe >= self.num_symbols] = self.pad_idx

            center_embed = self.symbol_emb(entity_ids_safe).unsqueeze(1)
            sim = F.cosine_similarity(center_embed, ent_embeds, dim=-1)
            topk = min(self.knn_k, sim.size(1))
            topk_vals, topk_idx = torch.topk(sim, k=topk, dim=-1)
            batch_idx = torch.arange(entities.size(0), device=entities.device).unsqueeze(1)
            rel_embeds = rel_embeds[batch_idx, topk_idx]
            ent_embeds = ent_embeds[batch_idx, topk_idx]

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)
        structural_repr = out.mean(dim=1).tanh()

        knn_mean = None
        if entity_ids is not None and self.knn_neighbors is not None:
            knn_idx = self.knn_neighbors[entity_ids_safe].to(self.symbol_emb.weight.device)
            knn_idx_safe = knn_idx.clone()
            knn_idx_safe[knn_idx_safe < 0] = self.pad_idx
            knn_idx_safe[knn_idx_safe >= self.num_symbols] = self.pad_idx

            knn_ent_embeds = self.dropout(self.symbol_emb(knn_idx_safe))
            center_embed = self.symbol_emb(entity_ids_safe).unsqueeze(1)
            sim = F.cosine_similarity(center_embed, knn_ent_embeds, dim=-1)
            topk = min(self.knn_k, sim.size(1))
            topk_vals, topk_idx = torch.topk(sim, k=topk, dim=-1)
            batch_idx = torch.arange(knn_idx_safe.size(0), device=knn_idx_safe.device).unsqueeze(1)
            knn_ent_embeds = knn_ent_embeds[batch_idx, topk_idx]

            pad_tensor = torch.full_like(knn_idx_safe, self.pad_idx, dtype=torch.long,
                                         device=self.symbol_emb.weight.device)
            knn_rel_embeds = self.dropout(self.symbol_emb(pad_tensor))

            concat_knn = torch.cat((knn_rel_embeds, knn_ent_embeds), dim=-1)
            out_knn = self.gcn_w(concat_knn)
            knn_mean = out_knn.mean(dim=1).tanh()

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

        if len(query_meta) == 6:
            q_l_conn, _, q_l_deg, q_r_conn, _, q_r_deg = query_meta
            s_l_conn, _, s_l_deg, s_r_conn, _, s_r_deg = support_meta
        else:
            q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
            s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta

        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        query_left = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids)
        query_right = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids)
        support_left = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)

        support_g = self.support_encoder(support_neighbor.unsqueeze(0))
        support_g = torch.mean(support_g, dim=1)
        query_g = self.support_encoder(query_neighbor.unsqueeze(1)).squeeze(1)
        query_f = self.query_encoder(support_g.squeeze(0), query_g)

        query_f = F.normalize(query_f, p=2, dim=-1)
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)

        matching_scores = torch.matmul(query_f, support_g.t()).squeeze(-1)
        return matching_scores

    def load_knn_index(self, knn_path):
        try:
            self.knn_neighbors = torch.load(knn_path)
            self.knn_k = self.knn_neighbors.shape[1]
            logging.info(f'LOADED k-NN: {self.knn_neighbors.shape}')
        except Exception as e:
            logging.error(f"Failed to load k-NN index: {e}")
