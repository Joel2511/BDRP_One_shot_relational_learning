import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from modules import *


class EmbedMatcher(nn.Module):
    """
    Universal Gating-Based Matcher for One-Shot Link Prediction.

    Supports:
        - NELL-One  : sparse graph, LUKE semantic embeddings, learned gate
        - FB15k-237 : dense graph, BERT embeddings, fixed gate (α=0.9 struct / 0.1 sem)
        - Medical   : hierarchical graph, PubMedBERT embeddings, learned gate
        - ATOMIC    : extreme sparse commonsense, RoBERTa embeddings, learned gate

    Key design choices:
        1. Relation-conditioned neighbor filtering:
           scores each (rel, ent) pair against the query relation direction,
           so topk selection is task-aware — critical for dense and hierarchical graphs.

        2. Max pooling after topk: one highly relevant neighbor drives the
           representation rather than being averaged out by irrelevant ones.

        3. DUAL-BRANCH GATE (learned mode):
           - Structural branch: GCN aggregation over neighborhood (connection matrix)
           - Semantic branch: direct raw embedding lookup for the center entity
             (this carries the LM signal: PubMedBERT/LUKE/RoBERTa)
           - Gate: sigmoid(linear(cat(structural, semantic))) → scalar alpha
           - Output: alpha * structural + (1-alpha) * semantic
           This replaces the previous broken gate that fed cat(x, x).

        4. gate_mode='fixed': pure structural, no gate parameters trained.
           Used for FB15k where a learned gate saturates.

    Parameters
    ----------
    gate_mode  : 'learned' | 'fixed'
    gate_alpha : float  (only used when gate_mode='fixed')
                 fraction of structural signal to keep (default 0.9)
    """

    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False,
                 aggregate='max', knn_k=32,
                 gate_mode='learned', gate_alpha=None):
        super(EmbedMatcher, self).__init__()

        # ---- embedding dimension ----
        self.actual_dim = embed.shape[1] if embed is not None else embed_dim
        self.embed_dim  = self.actual_dim

        # ---- gate config ----
        self.gate_mode  = gate_mode
        self.gate_alpha = gate_alpha if gate_alpha is not None else 0.9

        self.pad_idx     = num_symbols
        self.num_symbols = num_symbols
        self.knn_k       = knn_k
        self.aggregate   = aggregate

        # ---- embedding table ----
        self.symbol_emb = nn.Embedding(
            num_symbols + 1, self.actual_dim, padding_idx=num_symbols
        )

        # ---- GCN layer ----
        self.gcn_w = nn.Linear(2 * self.actual_dim, self.actual_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.actual_dim))

        # ---- relation-conditioned neighbor scoring projections ----
        self.rel_score_proj = nn.Linear(2 * self.actual_dim, self.actual_dim)
        self.rel_query_proj = nn.Linear(self.actual_dim, self.actual_dim)

        # ---- gate layer (only used when gate_mode='learned') ----
        # Takes cat(structural_repr, semantic_repr) → scalar alpha in [0, 1]
        # This is the DUAL-BRANCH gate — structural and semantic are genuinely different:
        #   structural = GCN aggregation over one-hop neighbors (topology signal)
        #   semantic   = raw embedding of center entity (LM signal)
        # The gate learns when to trust topology vs language model signal.
        self.gate_layer = nn.Linear(2 * self.actual_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # ---- init ----
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.rel_score_proj.weight)
        init.constant_(self.rel_score_proj.bias, 0)
        init.xavier_normal_(self.rel_query_proj.weight)
        init.constant_(self.rel_query_proj.bias, 0)
        # Gate bias init to 0 → initial alpha ≈ 0.5 (balanced blend at start of training)
        init.xavier_normal_(self.gate_layer.weight)
        init.constant_(self.gate_layer.bias, 0)

        # ---- pre-trained embeddings ----
        if use_pretrain and embed is not None:
            logging.info(f"LOADING {embed.shape[1]}D KB EMBEDDINGS INTO MATCHER")
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False

        # ---- matching network ----
        d_model = self.actual_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder   = QueryEncoder(d_model, process_steps)

    # ------------------------------------------------------------------
    def neighbor_encoder(self, connections, num_neighbors,
                         entity_ids=None, query_rel_emb=None):
        """
        Encode the one-hop neighborhood of a batch of entities.

        Args
        ----
        connections   : [B, max_neighbor, 2]  int — (relation_id, entity_id) per slot
        num_neighbors : [B]                   float — actual degree of each entity
        entity_ids    : [B]                   int — symbol IDs of the center entities
        query_rel_emb : [D]                   float — query relation embedding for
                                              relation-conditioned neighbor scoring.

        Returns
        -------
        [B, D]  — entity representation after neighborhood aggregation + gating
        """
        num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)

        relations  = connections[:, :, 0]   # [B, N]
        entities   = connections[:, :, 1]   # [B, N]

        rel_embeds = self.dropout(self.symbol_emb(relations))   # [B, N, D]
        ent_embeds = self.dropout(self.symbol_emb(entities))    # [B, N, D]

        # ---- relation-conditioned topk selection ----
        if entity_ids is not None:
            topk = min(self.knn_k, rel_embeds.size(1))

            if query_rel_emb is not None:
                # Score each (rel, ent) pair against the query relation direction.
                pair_emb  = torch.cat((rel_embeds, ent_embeds), dim=-1)           # [B, N, 2D]
                pair_proj = torch.tanh(self.rel_score_proj(pair_emb))              # [B, N, D]
                q_proj    = torch.tanh(self.rel_query_proj(query_rel_emb))         # [D]
                q_proj    = q_proj.unsqueeze(0).unsqueeze(0)                       # [1, 1, D]
                sim = F.cosine_similarity(
                    pair_proj, q_proj.expand_as(pair_proj), dim=-1
                )                                                                  # [B, N]
            else:
                # Fallback: plain entity-entity cosine
                center_embed = self.symbol_emb(entity_ids).unsqueeze(1)            # [B, 1, D]
                sim = F.cosine_similarity(center_embed, ent_embeds, dim=-1)        # [B, N]

            _, topk_idx = torch.topk(sim, k=topk, dim=-1)                         # [B, k]
            batch_idx   = torch.arange(
                entities.size(0), device=entities.device
            ).unsqueeze(1)
            rel_embeds  = rel_embeds[batch_idx, topk_idx]                          # [B, k, D]
            ent_embeds  = ent_embeds[batch_idx, topk_idx]                          # [B, k, D]

        # ---- GCN transform ----
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)                # [B, k, 2D]
        transformed   = torch.tanh(self.gcn_w(concat_embeds) + self.gcn_b)        # [B, k, D]

        # ---- max pooling ----
        structural_repr, _ = transformed.max(dim=1)                                # [B, D]

        # ---- gate (dataset-specific behaviour) ----
        if self.gate_mode == 'fixed':
            # Pure structural — no gate saturation risk.
            # For FB15k: topology signal is reliable, semantic is noisy.
            return structural_repr

        else:
            # DUAL-BRANCH learned gate:
            #   structural_repr = GCN aggregation (topology signal)
            #   semantic_repr   = raw embedding of center entity (LM signal)
            #
            # For Medical: PubMedBERT semantic + ComplEx structural are complementary.
            #   Gate learns: use structural for common ontology hubs,
            #                use semantic for rare leaf-node relations.
            #
            # For NELL-One: LUKE semantic + TransE structural are complementary.
            #   Sparse graph → neighbors are few → gate leans semantic when deg=0.
            #
            # For ATOMIC: structural is near-random (string entities, sparse graph).
            #   Gate should learn alpha≈0 (mostly semantic from RoBERTa).
            #   This happens naturally: structural_repr will be close to zero for
            #   isolated entities, and the gate will learn to suppress it.
            #
            # NOTE: entity_ids here are *symbol* IDs (not ent2id indices).
            # They index into symbol_emb which contains the LM-augmented embeddings.
            if entity_ids is not None:
                semantic_repr = self.symbol_emb(entity_ids)                        # [B, D]
                semantic_repr = self.dropout(semantic_repr)
            else:
                # Fallback: use structural as semantic if no entity_ids provided
                semantic_repr = structural_repr

            gate_input = torch.cat((structural_repr, semantic_repr), dim=-1)       # [B, 2D]
            alpha      = torch.sigmoid(self.gate_layer(gate_input))                # [B, 1]
            return alpha * structural_repr + (1.0 - alpha) * semantic_repr         # [B, D]

    # ------------------------------------------------------------------
    def forward(self, query, support, query_meta=None, support_meta=None):
        """
        Args
        ----
        query        : [Q, 2]  (head_symbol_id, tail_symbol_id)
        support      : [S, 2]
        query_meta   : (left_conn, left_deg, right_conn, right_deg)
        support_meta : same structure
        """
        if query_meta is None or support_meta is None:
            q_emb  = self.symbol_emb(query).view(query.size(0), -1)
            s_emb  = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))

        q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
        s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta

        q_h_ids = query[:, 0]
        q_t_ids = query[:, 1]
        s_h_ids = support[:, 0]
        s_t_ids = support[:, 1]

        # Query-relation direction from support head embeddings (same as G-Matching)
        query_rel_emb = self.symbol_emb(s_h_ids).mean(dim=0)   # [D]

        # Pass entity_ids (symbol IDs) to neighbor_encoder for dual-branch gate
        query_left    = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids, query_rel_emb)
        query_right   = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids, query_rel_emb)
        support_left  = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids, query_rel_emb)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids, query_rel_emb)

        query_neighbor   = torch.cat((query_left,   query_right),   dim=-1)       # [Q, 2D]
        support_neighbor = torch.cat((support_left, support_right), dim=-1)       # [S, 2D]

        support_g = torch.mean(
            self.support_encoder(support_neighbor.unsqueeze(0)), dim=1
        )                                                                          # [1, 2D]
        query_g   = self.support_encoder(
            query_neighbor.unsqueeze(1)
        ).squeeze(1)                                                               # [Q, 2D]

        query_f   = F.normalize(
            self.query_encoder(support_g.squeeze(0), query_g), p=2, dim=-1
        )
        support_g = F.normalize(support_g.squeeze(0), p=2, dim=-1)

        return torch.matmul(query_f, support_g.t()).squeeze(-1)
