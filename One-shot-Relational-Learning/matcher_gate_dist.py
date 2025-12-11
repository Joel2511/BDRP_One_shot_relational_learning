import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    1-Hop Gating + Distance Filtering.
    Replaces Attention weights with fixed weights derived from Cosine Similarity
    to ensure stability in the One-Shot setting.
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', 
                 gate_temp=1.0, k_neighbors=10): 
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.dropout = nn.Dropout(dropout)
        self.gate_temp = gate_temp
        self.k_neighbors = k_neighbors 

        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # --- Gating Components (Projection/Gate remains) ---
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))
        
        self.gate_w = nn.Linear(embed_dim, 1) 
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.gate_w.weight)
        init.constant_(self.gate_b, 0)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        d_model = embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors, entity_self_ids):
        relations = connections[:, :, 0]  # [B, K_max]
        entities = connections[:, :, 1]   # [B, K_max]
    
        # Embeddings (do not apply dropout before normalization used for cosine)
        rel_emb_raw = self.symbol_emb(relations)   # [B, K_max, D]
        ent_emb_raw = self.symbol_emb(entities)    # [B, K_max, D]
        self_emb_raw = self.symbol_emb(entity_self_ids).unsqueeze(1)  # [B, 1, D]
    
        # normalize first for stable cosine similarity
        norm_self = F.normalize(self_emb_raw, p=2, dim=-1)   # [B, 1, D]
        norm_neigh = F.normalize(ent_emb_raw, p=2, dim=-1)   # [B, K_max, D]
    
        # Cosine similarity scores between self and each neighbor entity
        similarity_scores = torch.bmm(norm_self, norm_neigh.transpose(1, 2)).squeeze(1)  # [B, K_max]
    
        is_pad = (relations == self.pad_idx)    # bool [B, K_max]
        similarity_scores = similarity_scores.masked_fill(is_pad, float('-1e9'))
    
        batch_size, max_k, dim = ent_emb_raw.shape
        k_to_select = min(self.k_neighbors, max_k)
        if k_to_select <= 0:
    
            self_emb = self.dropout(self_emb_raw).squeeze(1)  # [B, D]
            gate_input = self.gate_w(self_emb) + self.gate_b
            gate_val = torch.sigmoid(gate_input / (self.gate_temp + 1e-12))
            final_vec = gate_val * self_emb + (1.0 - gate_val) * self_emb  # effectively self_emb
            return torch.tanh(final_vec)
    
        # top-k indices and scores
        k_scores, k_indices = torch.topk(similarity_scores, k=k_to_select, dim=-1)  # [B, k]
        # Build boolean mask of selected indices per sample
        device = self.symbol_emb.weight.device
        k_mask = torch.zeros((batch_size, max_k), dtype=torch.bool, device=device)
        k_mask.scatter_(1, k_indices, True)  # True at selected positions
    
        rel_emb = self.dropout(rel_emb_raw)
        ent_emb = self.dropout(ent_emb_raw)
        self_emb = self.dropout(self_emb_raw).squeeze(1)  # [B, D]
    
        # GCN projection on all neighbors
        concat = torch.cat((rel_emb, ent_emb), dim=-1)  # [B, K_max, 2D]
        projected = self.gcn_w(concat) + self.gcn_b     # [B, K_max, D]
        projected = F.leaky_relu(projected)
    
        # Zero-out non-selected neighbors
        masked_projected = projected * k_mask.unsqueeze(-1).to(projected.dtype)  # [B, K_max, D]
    
    
        # Use the k_scores but convert them into positive weights via softmax across selected indices
        weights = torch.zeros((batch_size, max_k), device=device, dtype=projected.dtype)  # [B, K_max]
        # set weights for selected positions to the corresponding similarity score
        weights = weights.masked_scatter(k_mask, k_scores)  # places k_scores into selected positions
        # zero-out pad entries (they are already very negative in similarity but ensure zero weight)
        weights = weights.masked_fill(is_pad, 0.0)
        # softmax across neighbors to get normalized weights; add tiny epsilon to avoid all -inf
        weights = F.softmax(weights / (self.gate_temp + 1e-12), dim=1)  # [B, K_max]
        weights = weights.unsqueeze(-1)  # [B, K_max, 1]
    
        neighbor_agg = torch.sum(masked_projected * weights, dim=1)  # [B, D]
    
        # If an example had zero selected neighbors (all pads), weights would be uniform zeros -> sum 0.
        # In that case neighbor_agg is zero vector. Compute a fallback: if selected_count==0, use self_emb.
        selected_counts = k_mask.sum(dim=1)  # [B]
        zero_mask = (selected_counts == 0)
        if zero_mask.any():
            neighbor_agg[zero_mask] = self_emb[zero_mask]
    
        # Gating
        gate_input = self.gate_w(neighbor_agg) + self.gate_b  # [B, 1]
        gate_val = torch.sigmoid(gate_input / (self.gate_temp + 1e-12))  # [B, 1]
    
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_emb)
        return torch.tanh(final_vec)


    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta
        
        # --- ENCODE QUERY ---
        # Note: We are using a 1-Hop Gated Matcher here.
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1)

        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)
        
        # --- MATCHING NETWORK (Standard Baseline Pipeline) ---
        support_g = self.support_encoder(support_vec.unsqueeze(1)) 
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))
        
        # Mean Pooling of support set
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        
        support_g = support_g.squeeze(1) 
        query_encoded = query_encoded.squeeze(1) 
        
        # LSTM Query Refinement
        query_g = self.query_encoder(support_g, query_encoded)
        
        # Scoring
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
