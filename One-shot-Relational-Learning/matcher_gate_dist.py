import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import SupportEncoder, QueryEncoder


class EmbedMatcher(nn.Module):
    """
    1-Hop Gating + Distance Filtering matcher.

    Replaces attention weights with fixed weights derived from Cosine Similarity to
    ensure stability in a one-shot setting.

    Notes / fixes applied:
    - Proper imports and formatting
    - Safe pre-trained embedding loading (dtype/device aware)
    - Top-K selection mask built robustly
    - Division by actual selected neighbor count (avoids dividing by a fixed k when some
      neighbors are padding)
    - Minor typing and clearer docstrings
    """

    def __init__(
        self,
        embed_dim: int,
        num_symbols: int,
        use_pretrain: bool = True,
        embed: Optional[np.ndarray] = None,
        dropout: float = 0.2,
        batch_size: int = 64,
        process_steps: int = 4,
        finetune: bool = False,
        aggregate: str = "max",
        gate_temp: float = 1.0,
        k_neighbors: int = 10,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_symbols = num_symbols
        # padding index is the final index (num_symbols is number of real symbols)
        self.pad_idx = num_symbols
        self.aggregate = aggregate
        self.gate_temp = gate_temp
        self.k_neighbors = k_neighbors

        self.dropout = nn.Dropout(dropout)

        # +1 to allow a padding index
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # --- Gating / Projection components ---
        # Project concatenated (relation_emb || entity_emb) of size 2*embed_dim -> embed_dim
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # Gate: embed_dim -> 1
        self.gate_w = nn.Linear(embed_dim, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0.0)
        init.xavier_normal_(self.gate_w.weight)
        init.constant_(self.gate_b, 0.0)

        # Load pretrained symbol embeddings if requested
        if use_pretrain:
            logging.info("LOADING KB EMBEDDINGS")
            if embed is None:
                raise ValueError("use_pretrain=True but embed (numpy array) is None")
            # copy with proper dtype/device
            emb_tensor = torch.tensor(embed, dtype=self.symbol_emb.weight.dtype)
            # If emb_tensor shape matches (num_symbols + 1, embed_dim) or (num_symbols, embed_dim)
            if emb_tensor.shape[0] == num_symbols:
                # pad final row for padding idx
                pad_row = torch.zeros((1, emb_tensor.shape[1]), dtype=emb_tensor.dtype)
                emb_tensor = torch.cat([emb_tensor, pad_row], dim=0)
            if emb_tensor.shape != self.symbol_emb.weight.data.shape:
                raise ValueError(
                    f"Provided embed shape {emb_tensor.shape} does not match symbol_emb shape "
                    f"{self.symbol_emb.weight.data.shape}"
                )
            # copy to current device/dtype
            self.symbol_emb.weight.data.copy_(emb_tensor.to(self.symbol_emb.weight.device))

            if not finetune:
                logging.info("FREEZING KB EMBEDDING (no finetune)")
                self.symbol_emb.weight.requires_grad = False

        # Encoders use a doubled model dimension for concatenation of left/right
        d_model = embed_dim * 2
        # SupportEncoder and QueryEncoder are expected to be provided in modules
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(
        self,
        connections: torch.LongTensor,
        num_neighbors: torch.LongTensor,  # not used directly but kept for API compatibility
        entity_self_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Encode 1-hop neighbors for a batch of entities.

        Args:
            connections: LongTensor of shape [batch, K_max, 2] where connections[:,:,0] are relation ids
                         and connections[:,:,1] are entity ids.
            num_neighbors: (unused) LongTensor of shape [batch] or [batch, K_max] giving neighbor counts.
            entity_self_ids: LongTensor of shape [batch] giving the central entity id for each example.

        Returns:
            final_vec: Tensor of shape [batch, embed_dim] representing gated neighbor encoding.
        """
        # split relations / entities
        relations = connections[:, :, 0]  # [batch, K_max]
        entities = connections[:, :, 1]  # [batch, K_max]

        # embeddings
        rel_emb = self.dropout(self.symbol_emb(relations))  # [batch, K_max, dim]
        ent_emb = self.dropout(self.symbol_emb(entities))  # [batch, K_max, dim]
        self_emb = self.dropout(self.symbol_emb(entity_self_ids)).unsqueeze(1)  # [batch, 1, dim]

        batch_size, max_k, dim = ent_emb.shape

        # --- 1. Distance Calculation (Cosine similarity between center and each neighbor) ---
        norm_self = F.normalize(self_emb, p=2, dim=-1)  # [batch, 1, dim]
        norm_neigh = F.normalize(ent_emb, p=2, dim=-1)  # [batch, K_max, dim]
        # similarity: batch x 1 x K_max -> squeeze to batch x K_max
        similarity_scores = torch.bmm(norm_self, norm_neigh.transpose(1, 2)).squeeze(1)  # [batch, K_max]

        # Mask out padding neighbors (relations that equal pad_idx)
        is_pad = (relations == self.pad_idx).float()  # [batch, K_max]
        similarity_scores = similarity_scores - is_pad * 1e9

        # --- 2. Top-K Filtering (select up to self.k_neighbors neighbors per example) ---
        k_to_select = min(self.k_neighbors, max_k)
        # torch.topk requires k >= 1
        if k_to_select <= 0:
            raise ValueError("k_neighbors must be >= 1")

        # topk returns (values, indices)
        k_scores, k_indices = torch.topk(similarity_scores, k=k_to_select, dim=-1)  # [batch, k], [batch, k]
        # Build a boolean mask for selected indices
        current_device = self.symbol_emb.weight.device
        k_mask_float = torch.zeros((batch_size, max_k), device=current_device, dtype=torch.float32)
        # scatter 1.0 into selected positions
        k_mask_float.scatter_(1, k_indices, 1.0)
        k_mask = k_mask_float.unsqueeze(-1).bool()  # [batch, K_max, 1]

        # --- 3. GCN Projection & Aggregation ---
        concat = torch.cat((rel_emb, ent_emb), dim=-1)  # [batch, K_max, 2*dim]
        projected = self.gcn_w(concat) + self.gcn_b  # [batch, K_max, dim]
        projected = F.leaky_relu(projected)

        # zero-out not-selected neighbors
        filtered_projected = projected * k_mask.float()  # [batch, K_max, dim]

        # compute sum and divide by number of selected neighbors (per example)
        selected_counts = k_mask_float.sum(dim=1, keepdim=True)  # [batch, 1]
        neighbor_agg = filtered_projected.sum(dim=1) / (selected_counts + 1e-9)  # [batch, dim]

        # --- 4. Gating (softened by temperature) ---
        gate_input = self.gate_w(neighbor_agg) + self.gate_b  # [batch, 1]
        gate_val = torch.sigmoid(gate_input / self.gate_temp)  # [batch, 1]

        # 5. Residual connection with self embedding
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_emb.squeeze(1))  # [batch, dim]
        return torch.tanh(final_vec)

    def forward(
        self,
        query: torch.LongTensor,
        support: torch.LongTensor,
        query_meta: Optional[Tuple[torch.LongTensor, ...]] = None,
        support_meta: Optional[Tuple[torch.LongTensor, ...]] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: LongTensor [batch, 2] (h_id, t_id)
            support: LongTensor [batch_support, 2] (h_id, t_id)
            query_meta: metadata tuple for query neighbors. Expected layout:
                        (q_l1, _, q_deg_l, q_r1, _, q_deg_r)
                        where q_l1 and q_r1 are neighbor connection tensors shaped [batch, K_max, 2].
            support_meta: same layout for support set.
        Returns:
            scores: Tensor of similarity scores between refined query and support representation.
        """
        # Basic ids
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        if query_meta is None or support_meta is None:
            raise ValueError("query_meta and support_meta must be provided and follow expected layout")

        # Unpack metadata (keeps variable names from original code)
        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta

        # --- ENCODE QUERY (1-hop gated aggregator for left/right neighbors) ---
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1)  # [batch, 2*embed_dim]

        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)  # [batch_support, 2*embed_dim]

        # --- MATCHING NETWORK (Standard Baseline Pipeline) ---
        # Encode support and query via the same support_encoder (unsqueeze for sequence dim)
        support_g = self.support_encoder(support_vec.unsqueeze(1))  # shape depends on SupportEncoder internals
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))

        # Mean-pool over support set (original code used mean over dimension 0)
        support_g = torch.mean(support_g, dim=0, keepdim=True)  # keepdim True for safe squeeze
        support_g = support_g.squeeze(1)  # [batch_support?, embed]
        query_encoded = query_encoded.squeeze(1)

        # LSTM / iterative query refinement via the query_encoder
        query_g = self.query_encoder(support_g, query_encoded)  # expected output shapes align

        # Final scoring (dot product)
        scores = torch.sum(query_g * support_g, dim=1)

        return scores
