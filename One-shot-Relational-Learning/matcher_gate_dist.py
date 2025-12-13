import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    1-Hop Gating + Distance Filtering.
    Incorporates 'Joint Cosine Similarity' for ComplEx via standard cosine on concatenated tensors.
    Uses Residual Gating and Dynamic Averaging to fix Long-Tail/Zero-Value issues.
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', 
                 k_neighbors=10,gate_temp=1.0): 
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.dropout = nn.Dropout(dropout)
        self.k_neighbors = k_neighbors 

        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # --- Gating Components ---
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))
        
        self.gate_w = nn.Linear(embed_dim, 1) 
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Learnable Temperature (Initialized to 1.0)
        self.gate_temp = nn.Parameter(torch.tensor(1.0))

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
        """
        Encodes 1-Hop neighbors using Distance Filtering (Top-K) + Soft Residual Gating.
        """
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        
        # 1. Get RAW Embeddings (No Dropout yet) for stable Distance Calculation
        # This implements "Joint Cosine Similarity" for ComplEx automatically
        rel_emb_raw = self.symbol_emb(relations)
        ent_emb_raw = self.symbol_emb(entities)
        self_emb_raw = self.symbol_emb(entity_self_ids).unsqueeze(1)

        batch_size, max_k, dim = ent_emb_raw.shape
        
        # --- Distance Filtering ---
        # Normalize raw embeddings to get Cosine Similarity
        norm_self = F.normalize(self_emb_raw, p=2, dim=-1) 
        norm_neigh = F.normalize(ent_emb_raw, p=2, dim=-1)
        
        # Calculate Similarity: (B, 1, D) x (B, D, K) -> (B, 1, K) -> (B, K)
        similarity_scores = torch.bmm(norm_self, norm_neigh.transpose(1, 2)).squeeze(1) 

        # Mask padding (relation_id is PAD)
        is_pad = (relations == self.pad_idx).float() 
        similarity_scores = similarity_scores - (is_pad * 1e9) # Push pads to -inf

        # Select Top-K
        # If an entity has 0 neighbors, this will select pads, but we handle that next
        k_to_select = min(self.k_neighbors, max_k)
        k_scores, k_indices = torch.topk(similarity_scores, k=k_to_select, dim=-1)
        
        # Create Selection Mask
        current_device = self.symbol_emb.weight.device
        k_mask = torch.zeros((batch_size, max_k), device=current_device)
        k_mask.scatter_(1, k_indices, 1)
        k_mask = k_mask.unsqueeze(-1) # Broadcastable mask

        # --- Aggregation ---
        # NOW we apply dropout to features we want to use for training
        rel_emb = self.dropout(rel_emb_raw)
        ent_emb = self.dropout(ent_emb_raw)
        
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      
        projected = self.gcn_w(concat) + self.gcn_b         
        projected = F.leaky_relu(projected)

        # Apply Top-K Mask
        filtered_projected = projected * k_mask
        
        # Valid Mask: Ensure we don't count pads even if they were in Top-K (edge case)
        valid_mask = (1.0 - is_pad.unsqueeze(-1)) * k_mask
        
        # CRITICAL FIX: Dynamic Denominator
        # Divide by the ACTUAL number of valid neighbors selected, not K.
        # Clamp to 1.0 to avoid division by zero for isolated nodes.
        denom = valid_mask.sum(dim=1)
        denom = torch.clamp(denom, min=1.0)
        
        neighbor_agg = torch.sum(filtered_projected * valid_mask, dim=1) / denom

        # --- Soft Gating (Residual) ---
        # Learnable temperature
        gate_input = self.gate_w(neighbor_agg) + self.gate_b
        actual_temp = torch.clamp(self.gate_temp, min=0.01, max=10.0) # Prevent extremes
        gate_val = torch.sigmoid(gate_input / actual_temp) 
        
        # RESIDUAL CONNECTION: Self + (Gate * Neighbor)
        # This ensures the Self embedding is never suppressed.
        self_emb_final = self.dropout(self.symbol_emb(entity_self_ids)) # Apply dropout to self here
        final_vec = self_emb_final + (gate_val * neighbor_agg)

        return torch.tanh(final_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta
        
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1)

        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)
        
        support_g = self.support_encoder(support_vec.unsqueeze(1)) 
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))
        
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        support_g = support_g.squeeze(1) 
        query_encoded = query_encoded.squeeze(1) 
        
        query_g = self.query_encoder(support_g, query_encoded)
        
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
