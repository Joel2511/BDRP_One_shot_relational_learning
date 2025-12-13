import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    Distance-Based Neighbor Encoding + Content-Based Soft Gating.
    Optimized for Dense Graphs (ATOMIC / FB15k).
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', 
                 k_neighbors=10):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.k_neighbors = k_neighbors
        self.dropout = nn.Dropout(dropout)
        
        # Learnable Temperature
        # Starts at 1.0. The optimizer will learn the best sharpness.
        self.gate_temp = nn.Parameter(torch.tensor(1.0)) 

        # Embedding layer
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # Neighbor projection (GCN-style)
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # --- Content-Based Gate (MLP) ---
        # Decides how much to trust the neighbors based on the vector content.
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        
        for m in self.context_gate.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

        if use_pretrain and embed is not None:
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
        1. Selects Top-K neighbors based on Semantic Distance (Cosine).
        2. Aggregates them.
        3. Gates the result using a Learnable Soft Gate.
        """
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        
        # 1. Get RAW Embeddings (No Dropout for Distance Calculation)
        rel_emb_raw = self.symbol_emb(relations)
        ent_emb_raw = self.symbol_emb(entities)
        self_emb_raw = self.symbol_emb(entity_self_ids).unsqueeze(1) # [B, 1, D]

        batch_size, max_k, dim = ent_emb_raw.shape
        
        # --- Distance Filtering (Metric Learning) ---
        # Normalize for Cosine Similarity
        norm_self = F.normalize(self_emb_raw, p=2, dim=-1) 
        norm_neigh = F.normalize(ent_emb_raw, p=2, dim=-1)
        
        # Similarity: (B, 1, D) x (B, D, K) -> (B, K)
        # Higher score = Closer distance
        similarity_scores = torch.bmm(norm_self, norm_neigh.transpose(1, 2)).squeeze(1) 

        # Mask padding (relation_id is PAD)
        is_pad = (relations == self.pad_idx).float() 
        similarity_scores = similarity_scores - (is_pad * 1e9) 

        # Select Top-K Neighbors
        k_to_select = min(self.k_neighbors, max_k)
        k_scores, k_indices = torch.topk(similarity_scores, k=k_to_select, dim=-1)
        
        # Create Selection Mask
        current_device = self.symbol_emb.weight.device
        k_mask = torch.zeros((batch_size, max_k), device=current_device)
        k_mask.scatter_(1, k_indices, 1.0)
        k_mask = k_mask.unsqueeze(-1) # [B, K, 1]

        # --- Aggregation ---
        # Now apply dropout for feature learning
        rel_emb = self.dropout(rel_emb_raw)
        ent_emb = self.dropout(ent_emb_raw)
        
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      
        projected = self.gcn_w(concat) + self.gcn_b         
        projected = F.leaky_relu(projected)

        # Apply Top-K Mask
        filtered_projected = projected * k_mask
        
        # Dynamic Denominator: Count valid neighbors (non-pad AND selected)
        valid_mask = (1.0 - is_pad.unsqueeze(-1)) * k_mask
        denom = valid_mask.sum(dim=1)
        denom = torch.clamp(denom, min=1.0) # Avoid div/0
        
        neighbor_agg = torch.sum(filtered_projected * valid_mask, dim=1) / denom

        # --- Soft Content Gating ---
        # "Is this aggregated neighbor vector useful?"
        gate_logit = self.context_gate(neighbor_agg) # [B, 1]
        
        # Clamp temp for stability
        curr_temp = torch.clamp(self.gate_temp, min=0.1, max=5.0)
        gate_val = torch.sigmoid(gate_logit / curr_temp)
        
        # Safety check: If node is isolated, gate must be 0
        has_neighbors = (num_neighbors > 0).float().unsqueeze(1)
        gate_val = gate_val * has_neighbors

        # Residual Connection: Self + (Gate * Neighbor)
        self_emb_final = self.dropout(self.symbol_emb(entity_self_ids))
        final_vec = self_emb_final + (gate_val * neighbor_agg)

        return torch.tanh(final_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        # Flexible Unpacking to prevent crashes
        if len(query_meta) == 6:
             q_l1, q_deg_l, q_r1, q_deg_r = query_meta[0], query_meta[2], query_meta[3], query_meta[5]
             s_l1, s_deg_l, s_r1, s_deg_r = support_meta[0], support_meta[2], support_meta[3], support_meta[5]
        else:
             q_l1, q_deg_l, q_r1, q_deg_r = query_meta
             s_l1, s_deg_l, s_r1, s_deg_r = support_meta
        
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
