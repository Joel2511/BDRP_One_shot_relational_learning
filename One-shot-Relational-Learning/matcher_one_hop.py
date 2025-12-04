import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    1-Hop Gated Attention Network.
    Applies Gating and Attention to 1-Hop neighbors only.
    """import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    1-Hop Gated Attention Network (Corrected Pipeline).
    Matches Baseline's query processing exactly, but uses Gated Attention for neighbors.
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.dropout = nn.Dropout(dropout)

        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # --- Gated Attention Components ---
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))
        self.attn_linear = nn.Linear(embed_dim, 1)
        self.gate_w = nn.Linear(embed_dim, 1) 
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.attn_linear.weight)
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
        Encodes 1-Hop neighbors using Attention and Gating.
        """
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        
        rel_emb = self.dropout(self.symbol_emb(relations)) 
        ent_emb = self.dropout(self.symbol_emb(entities))  
        self_emb = self.dropout(self.symbol_emb(entity_self_ids)).unsqueeze(1) 

        # GCN Projection
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      
        projected = self.gcn_w(concat) + self.gcn_b         
        projected = F.leaky_relu(projected) # Added non-linearity like GANA

        # Attention
        attn_scores = self.attn_linear(projected)           
        attn_weights = F.softmax(attn_scores, dim=1)        
        neighbor_agg = torch.sum(projected * attn_weights, dim=1) 

        # Gating (Noise Filter)
        gate_val = torch.sigmoid(self.gate_w(neighbor_agg) + self.gate_b)
        
        # Residual Connection (Gate * Neigh + (1-Gate) * Self)
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_emb.squeeze(1))

        # [NOTE] Baseline uses .tanh() here. We can add it for stability.
        return torch.tanh(final_vec) 

    def forward(self, query, support, query_meta=None, support_meta=None):
        # Extract IDs
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta
        
        # --- 1. Encode Neighbors (Gated Attention) ---
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1) # [Batch, 2*Dim]

        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1) # [Batch, 2*Dim]
        
        # --- 2. Matching Network (CORRECTED) ---
        
        # A. Encode Support
        # SupportEncoder expects [Batch, Seq, Dim]. We have [Batch, Dim].
        # We unsqueeze to make it a sequence of length 1.
        support_g = self.support_encoder(support_vec.unsqueeze(1)) 
        
        # B. Encode Query (THE MISSING STEP!)
        # We must also pass the query through the SupportEncoder to project/normalize it.
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))
        
        # C. Mean Pooling (For support set)
        # Even with 1-shot, this keeps dimensions consistent
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        
        # D. Squeeze back to 2D for LSTM
        support_g = support_g.squeeze(1) # [1, Dim] (Broadcasting happens inside QueryEncoder)
        query_encoded = query_encoded.squeeze(1) # [Batch, Dim]
        
        # E. LSTM Query Refinement
        # QueryEncoder(support, query)
        query_g = self.query_encoder(support_g, query_encoded)
        
        # F. Score
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.dropout = nn.Dropout(dropout)

        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # --- Gated Attention Components ---
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))
        self.attn_linear = nn.Linear(embed_dim, 1)
        self.gate_w = nn.Linear(embed_dim, 1) 
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.attn_linear.weight)
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
        Encodes 1-Hop neighbors using Attention and Gating.
        """
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        
        rel_emb = self.dropout(self.symbol_emb(relations)) 
        ent_emb = self.dropout(self.symbol_emb(entities))  
        self_emb = self.dropout(self.symbol_emb(entity_self_ids)).unsqueeze(1) 

        # GCN Projection
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      
        projected = self.gcn_w(concat) + self.gcn_b         
        projected = F.leaky_relu(projected)

        # Attention
        attn_scores = self.attn_linear(projected)           
        attn_weights = F.softmax(attn_scores, dim=1)        
        neighbor_agg = torch.sum(projected * attn_weights, dim=1) 

        # Gating
        gate_val = torch.sigmoid(self.gate_w(neighbor_agg) + self.gate_b)
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_emb.squeeze(1))

        return final_vec

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta
        
        # --- ONLY ENCODE 1-HOP ---
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1)

        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)
        
        # Matching Network
        support_g = self.support_encoder(support_vec) 
        query_g = self.query_encoder(support_g, query_vec)
        
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
