import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    Robust One-Shot Matcher with Content-Based Gating.
    
    1. Similarity Attention: Weights neighbors by relevance to center (Dot Product).
    2. Content-Based Gating: An MLP judges signal quality directly from the data.
       Crucially, this generalizes to UNSEEN relations (Zero-Shot/One-Shot compatible).
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', 
                 k_neighbors=15): # Default K=15 based on your script
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.k_neighbors = k_neighbors
        self.dropout = nn.Dropout(dropout)
        
        # Learnable Temperature
        # Starts at 1.0. Optimizer will adjust it.
        self.gate_temp = nn.Parameter(torch.tensor(1.0)) 

        # Embedding layer
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # Neighbor projection (GCN-style)
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # --- FIX 1: Content-Based Gate ---
        # Instead of nn.Embedding (which fails for new relations), we use an MLP 
        # to judge the quality of the neighbor aggregation itself.
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
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        
        # 1. Embeddings (Raw for similarity calculation)
        rel_emb_raw = self.symbol_emb(relations)
        ent_emb_raw = self.symbol_emb(entities)
        self_emb = self.symbol_emb(entity_self_ids).unsqueeze(1) # [B, 1, D]

        # 2. Project Neighbors (GCN Step)
        concat_emb = torch.cat((rel_emb_raw, ent_emb_raw), dim=-1)
        projected = self.gcn_w(concat_emb) + self.gcn_b
        projected = F.leaky_relu(projected) 
        
        # Apply dropout to features used for aggregation
        projected = self.dropout(projected)

        # 3. Similarity Attention 
        # Calculate relevance of each neighbor to the center
        # (B, K, D) * (B, 1, D) -> sum -> (B, K)
        attn_scores = (projected * self_emb).sum(dim=-1) 
        
        # Mask padding
        is_pad = (relations == self.pad_idx)
        attn_scores = attn_scores.masked_fill(is_pad, -1e9)
        
        # Weights
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1) # [B, K, 1]
        
        # Weighted Sum
        neighbor_agg = (attn_weights * projected).sum(dim=1) # [B, D]

        # 4. Input-Dependent Soft Gating (The Fix)
        # Determine trust based on the aggregated signal itself
        gate_logit = self.context_gate(neighbor_agg) # [B, 1]
        
        # Clamp temp to avoid instability
        curr_temp = torch.clamp(self.gate_temp, min=0.1, max=5.0)
        gate_val = torch.sigmoid(gate_logit / curr_temp)
        
        # Zero-out gate if node has absolutely no neighbors
        has_neighbors = (num_neighbors > 0).float().unsqueeze(1)
        gate_val = gate_val * has_neighbors

        # 5. Residual Connection
        # Self + (Gate * Neighbors). Ensures rare entities retain their identity.
        final_vec = self_emb.squeeze(1) + (gate_val * neighbor_agg)

        return torch.tanh(final_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        # --- FIX: Flexible Unpacking ---
        # The trainer might send 4 items (1-hop) or 6 items (2-hop with placeholders).
        # We explicitly grab what we need based on tuple size.
        if len(query_meta) == 6:
            # 2-hop format: (l1, l2, deg_l, r1, r2, deg_r)
            # We ignore index 1 and 4 (the 2-hop connections)
            q_l1 = query_meta[0]
            q_deg_l = query_meta[2]
            q_r1 = query_meta[3]
            q_deg_r = query_meta[5]
            
            s_l1 = support_meta[0]
            s_deg_l = support_meta[2]
            s_r1 = support_meta[3]
            s_deg_r = support_meta[5]
        else:
            # 1-hop format: (l1, deg_l, r1, deg_r)
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
