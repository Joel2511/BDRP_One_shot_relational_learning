import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    1-Hop Gating ONLY (No Attention), with Temperature Control.
    Logic: Gate * Mean(Neighbors) + (1-Gate) * Self
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', 
                 gate_temp=1.0): #Temperature parameter (Default 1.0)
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.dropout = nn.Dropout(dropout)
        self.gate_temp = gate_temp # Stores the temperature

        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # --- Gating Components ---
        # Projection layer
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))
        
        # Gate weights
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
        """
        Encodes 1-Hop neighbors using Mean Aggregation + Temperature-Softened Gating.
        """
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        
        rel_emb = self.dropout(self.symbol_emb(relations)) 
        ent_emb = self.dropout(self.symbol_emb(entities))  
        self_emb = self.dropout(self.symbol_emb(entity_self_ids)).unsqueeze(1) 

        # 1. GCN Projection
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      
        projected = self.gcn_w(concat) + self.gcn_b         
        projected = F.leaky_relu(projected)

        # 2. Mean Aggregation (Like Baseline)
        # Sum neighbors and divide by count (handling padding) or just mean over dim 1
        neighbor_agg = torch.mean(projected, dim=1) 

        # 3. Gating (Softened by Temperature)
        # We divide the input to sigmoid by the temperature self.gate_temp
        # If temp > 1, the curve flattens (softer). If temp < 1, it sharpens.
        gate_input = self.gate_w(neighbor_agg) + self.gate_b
        gate_val = torch.sigmoid(gate_input / self.gate_temp) 
        
        # Residual Connection
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_emb.squeeze(1))

        return torch.tanh(final_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        # Assuming the trainer handles the split correctly (q_l1, None, q_deg_l, ...)
        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta
        
        # --- ENCODE QUERY ---
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1)

        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)
        
        # --- MATCHING NETWORK (Standard Baseline Pipeline) ---
        
        # 1. Project Support
        support_g = self.support_encoder(support_vec.unsqueeze(1)) 
        
        # 2. Project Query (Ensures query is in same space as support)
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))
        
        # 3. Mean Pooling of support set (Standard Baseline stability trick)
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        
        # 4. Squeeze for LSTM
        support_g = support_g.squeeze(1) 
        query_encoded = query_encoded.squeeze(1) 
        
        # 5. LSTM Query Refinement
        query_g = self.query_encoder(support_g, query_encoded)
        
        # 6. Scoring
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
