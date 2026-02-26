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
    Key improvements over original:
      - Relation-conditioned neighbor filtering (not purely entity-entity cosine)
      - Max pooling instead of mean pooling after topk (better for dense graphs)
      - Semantic signal lives in symbol_emb (BERT-concat at load time); gate is kept
        but only activates when knn_neighbors is actually provided
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False,
                 aggregate='max', knn_k=32):
        super(EmbedMatcher, self).__init__()

        self.actual_dim = embed.shape[1] if embed is not None else embed_dim
        self.embed_dim = self.actual_dim

        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.actual_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols
        self.knn_k = knn_k

        # GCN weights
        self.gcn_w = nn.Linear(2 * self.actual_dim, self.actual_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.actual_dim))

        # Relation-conditioned scoring projection: maps (rel+ent concat) to scalar score
        # used to rank neighbors by relevance to query relation
        self.rel_score_proj = nn.Linear(2 * self.actual_dim, self.actual_dim)
        self.rel_query_proj = nn.Linear(self.actual_dim, self.actual_dim)

        # Sigmoid Gating Layer (structural vs semantic)
        self.gate_layer = nn.Linear(2 * self.actual_dim, 1)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.rel_score_proj.weight)
        init.xavier_normal_(self.rel_query_proj.weight)
        init.xavier_normal_(self.gate_layer.weight)
        init.constant_(self.gate_layer.bias, 0)

        if use_pretrain and embed is not None:
            logging.info(f'LOADING {embed.shape[1]}D KB EMBEDDINGS INTO MATCHER')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False

        d_model = self.actual_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None, query_rel_emb=None):
        """
        Encode neighbors of a batch of entities.

        Args:
            connections:    [B, max_neighbor, 2]  (relation_id, entity_id)
            num_neighbors:  [B]
            entity_ids:     [B]  symbol IDs of the center entities (for filtering)
            query_rel_emb:  [D]  embedding of the query relation — used for
                                 relation-conditioned neighbor scoring (key fix for FB15k)
        """
        num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)
        relations = connections[:, :, 0]   # [B, N]
        entities  = connections[:, :, 1]   # [B, N]

        rel_embeds = self.dropout(self.symbol_emb(relations))   # [B, N, D]
        ent_embeds = self.dropout(self.symbol_emb(entities))     # [B, N, D]

        # ------------------------------------------------------------------ #
        #  Relation-conditioned neighbor filtering                            #
        #  Score each (rel, ent) pair by how relevant it is to the query      #
        #  relation, then keep the top-k.                                     #
        #  Falls back to entity-entity cosine when no query_rel_emb given.   #
        # ------------------------------------------------------------------ #
        if entity_ids is not None:
            topk = min(self.knn_k, rel_embeds.size(1))

            if query_rel_emb is not None:
                # Project (rel+ent) pairs into a common space and score against
                # the projected query relation embedding.
                pair_emb = torch.cat((rel_embeds, ent_embeds), dim=-1)          # [B, N, 2D]
                pair_proj = torch.tanh(self.rel_score_proj(pair_emb))            # [B, N, D]

                # Query relation projected to same space; broadcast over batch
                q_proj = torch.tanh(self.rel_query_proj(query_rel_emb))          # [D]
                q_proj = q_proj.unsqueeze(0).unsqueeze(0)                        # [1, 1, D]

                sim = F.cosine_similarity(pair_proj, q_proj.expand_as(pair_proj), dim=-1)  # [B, N]
            else:
                # Fallback: entity-entity similarity (original behaviour)
                center_embed = self.symbol_emb(entity_ids).unsqueeze(1)          # [B, 1, D]
                sim = F.cosine_similarity(center_embed, ent_embeds, dim=-1)      # [B, N]

            _, topk_idx = torch.topk(sim, k=topk, dim=-1)                       # [B, k]
            batch_idx  = torch.arange(entities.size(0), device=entities.device).unsqueeze(1)
            rel_embeds = rel_embeds[batch_idx, topk_idx]                         # [B, k, D]
            ent_embeds = ent_embeds[batch_idx, topk_idx]                         # [B, k, D]

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)              # [B, k, 2D]
        transformed   = torch.tanh(self.gcn_w(concat_embeds) + self.gcn_b)      # [B, k, D]

        # ------------------------------------------------------------------ #
        #  Max pooling — much better than mean on dense graphs because a     #
        #  single highly-informative neighbor can dominate rather than being  #
        #  averaged out by many irrelevant ones.                              #
        # ------------------------------------------------------------------ #
        structural_repr, _ = transformed.max(dim=1)                              # [B, D]

        return structural_repr

    def forward(self, query, support, query_meta=None, support_meta=None):
        """
        query:        [Q, 2]  (head_symbol_id, tail_symbol_id)
        support:      [S, 2]
        query_meta:   (left_conn, left_deg, right_conn, right_deg)
        support_meta: same structure
        """
        if query_meta is None or support_meta is None:
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))

        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
        s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta

        # ------------------------------------------------------------------ #
        #  Extract a query-relation embedding to condition neighbor scoring.  #
        #  We use the mean of support head embeddings as a proxy for the      #
        #  relation direction (same trick as the original G-Matching paper).  #
        # ------------------------------------------------------------------ #
        query_rel_emb = self.symbol_emb(s_h_ids).mean(dim=0)                    # [D]

        query_left    = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids, query_rel_emb)
        query_right   = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids, query_rel_emb)
        support_left  = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids, query_rel_emb)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids, query_rel_emb)

        query_neighbor   = torch.cat((query_left,   query_right),   dim=-1)     # [Q, 2D]
        support_neighbor = torch.cat((support_left, support_right), dim=-1)     # [S, 2D]

        support_g = torch.mean(self.support_encoder(support_neighbor.unsqueeze(0)), dim=1)
        query_g   = self.support_encoder(query_neighbor.unsqueeze(1)).squeeze(1)

        query_f   = F.normalize(self.query_encoder(support_g.squeeze(0), query_g), p=2, dim=-1)
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)

        return torch.matmul(query_f, support_g.t()).squeeze(-1)
