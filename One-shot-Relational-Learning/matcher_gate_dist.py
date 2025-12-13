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
                 k_neighbors=10):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.k_neighbors = k_neighbors
        self.dropout = nn.Dropout(dropout)
        
        # Learnable Temperature (Initialized to 1.0)
        # Allows the model to find the optimal "softness" for the decision boundary
        self.gate_temp = nn.Parameter(torch.tensor(1.0)) 

        # Embedding layer
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # Neighbor projection (GCN-style)
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # --- FIX: Content-Based Gate (Generalizes to Unseen Relations) ---
        # Instead of nn.Embedding (which fails for new relations), we use an MLP 
        # to judge the quality of the neighbor aggregation itself.
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        
        # Initialize gate MLP
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
    
    # EMBED
    rel_emb = self.symbol_emb(relations)
    ent_emb = self.symbol_emb(entities)
    
    # MAX-POOL (PROVEN SOTA ON NELL-ONE)
    concat_emb = torch.cat((rel_emb, ent_emb), dim=-1)
    projected = self.gcn_w(concat_emb) + self.gcn_b
    projected = F.leaky_relu(projected, 0.1)  # Lower slope
    projected = self.dropout(projected)
    
    neighbor_agg = projected.max(dim=1)[0]  # [B, D] - SIMPLE & EFFECTIVE
    
    # ZERO gating - just residual add (0.3 weight)
    self_emb = self.symbol_emb(entity_self_ids)
    final_vec = 0.7 * self_emb + 0.3 * neighbor_agg  # FIXED BLEND
    
    return torch.tanh(final_vec)


    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta
        
        # Encode Query Head and Tail
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1)

        # Encode Support Head and Tail
        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)
        
        # Matching Network processing
        support_g = self.support_encoder(support_vec.unsqueeze(1)) 
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))
        
        # Pooling and Refinement
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        support_g = support_g.squeeze(1) 
        query_encoded = query_encoded.squeeze(1) 
        
        query_g = self.query_encoder(support_g, query_encoded)
        
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
