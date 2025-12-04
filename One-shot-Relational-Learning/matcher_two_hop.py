import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Assuming SupportEncoder and QueryEncoder are defined in modules.py
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings with GANA-style 2-hop Gating + Attention.
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Symbol Embeddings (Entities + Relations)
        self.symbol_emb = nn.Embedding(
            num_symbols + 1,
            embed_dim,
            padding_idx=self.pad_idx
        )

        # --- GANA COMPONENTS ---
        
        # 1. Neighbor Projection (GCN)
        # Maps [Relation; Entity] -> Hidden Dim
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # 2. Attention Scorer (To weight neighbors)
        # Scores importance of each neighbor
        self.attn_linear = nn.Linear(embed_dim, 1)

        # 3. Gating Mechanism (The "Noise Filter")
        # Calculates the gate value alpha = sigmoid(W * neighbor_agg + b)
        self.gate_w = nn.Linear(embed_dim, 1) 
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # --- Initialization ---
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.attn_linear.weight)
        init.xavier_normal_(self.gate_w.weight)
        init.constant_(self.gate_b, 0)

        # Load Pre-trained Embeddings
        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        # Encoders for the Matching Network
        d_model = embed_dim * 2  # Because we represent pairs (Head+Tail) or (Head+Neighbor)
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors, entity_self_ids):
        """
        Encodes neighbors using GANA logic:
        1. Project Neighbors
        2. LeakyReLU Activation (Added to match paper)
        3. Attention Aggregation
        4. Gating (Neighbor Aggregation vs Self Embedding)
        """
        # 1. Retrieve Embeddings
        # connections[:, :, 0] is Relation, [:, :, 1] is Tail Entity
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]

        rel_emb = self.dropout(self.symbol_emb(relations)) # (Batch, Neigh, Dim)
        ent_emb = self.dropout(self.symbol_emb(entities))  # (Batch, Neigh, Dim)
        
        # Get "Self" embedding for the Residual Connection
        # Shape: (Batch, 1, Dim)
        self_emb = self.dropout(self.symbol_emb(entity_self_ids)).unsqueeze(1) 

        # 2. GCN Projection (Relation + Entity -> Feature)
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      # (Batch, Neigh, 2*Dim)
        projected = self.gcn_w(concat) + self.gcn_b         # (Batch, Neigh, Dim)
        
        # [CRITICAL CHANGE] GANA applies LeakyReLU before Attention
        projected = F.leaky_relu(projected)

        # 3. Attention Mechanism
        # Calculate attention scores
        attn_scores = self.attn_linear(projected)           # (Batch, Neigh, 1)
        
        # Apply Softmax to get weights (Values sum to 1 across neighbors)
        attn_weights = F.softmax(attn_scores, dim=1)        # (Batch, Neigh, 1)
        
        # Weighted Sum of neighbors
        neighbor_agg = torch.sum(projected * attn_weights, dim=1) # (Batch, Dim)

        # 4. Gating (The "Noise Filter")
        # Compute gate value between 0 and 1 based on the aggregated context
        gate_val = torch.sigmoid(self.gate_w(neighbor_agg) + self.gate_b) # (Batch, 1)
        
        # 5. Residual Connection with Gating
        # If Gate is 1, we trust the Neighbors.
        # If Gate is 0, we ignore Neighbors and use Self (Baseline behavior).
        self_vec = self_emb.squeeze(1)
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_vec)

        return torch.tanh(final_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        """
        Args:
            query: (Batch, 2) - [Head_ID, Tail_ID] for the query
            support: (Batch, 2) - [Head_ID, Tail_ID] for the support set (one-shot)
            query_meta: Tuple containing neighbor adjacency for query entities
            support_meta: Tuple containing neighbor adjacency for support entities
        """
        # Fallback for Baseline mode (if no meta provided)
        if (query_meta is None) or (support_meta is None):
            return self.run_baseline_logic(query, support)
        
        # Extract Self IDs from the main input pairs (Head, Tail)
        q_h_ids = query[:, 0]
        q_t_ids = query[:, 1]
        s_h_ids = support[:, 0]
        s_t_ids = support[:, 1]

        # Unpack Meta Data
        # Format: (Left_Hop1, Left_Hop2, Deg_L, Right_Hop1, Right_Hop2, Deg_R)
        (q_l1, q_l2, q_deg_l, q_r1, q_r2, q_deg_r) = query_meta
        (s_l1, s_l2, s_deg_l, s_r1, s_r2, s_deg_r) = support_meta
        
        # --- ENCODE QUERY ---
        # 1. Encode 1-Hop Neighbors (Self is q_h_ids)
        q_h_1 = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_t_1 = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        
        # 2. Encode 2-Hop Neighbors (Self is STILL q_h_ids)
        # Treating 2-hop neighbors as context for the *same* center entity
        q_h_2 = self.neighbor_encoder(q_l2, None, q_h_ids)
        q_t_2 = self.neighbor_encoder(q_r2, None, q_t_ids)

        # --- ENCODE SUPPORT ---
        s_h_1 = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_t_1 = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        
        s_h_2 = self.neighbor_encoder(s_l2, None, s_h_ids)
        s_t_2 = self.neighbor_encoder(s_r2, None, s_t_ids)

        # --- FUSE HOPS ---
        # Average the gated representations from both hops
        q_left = (q_h_1 + q_h_2) / 2.0
        q_right = (q_t_1 + q_t_2) / 2.0
        
        s_left = (s_h_1 + s_h_2) / 2.0
        s_right = (s_t_1 + s_t_2) / 2.0

        # Construct Pair Representations (Head + Tail)
        query_vec = torch.cat((q_left, q_right), dim=-1)     # (Batch, 2*Dim)
        support_vec = torch.cat((s_left, s_right), dim=-1)   # (Batch, 2*Dim)
        
        # --- MATCHING NETWORK ---
        # 1. Encode Support
        # SupportEncoder expects [Batch, Seq, Dim] or [Batch, Dim]
        # We unsqueeze to mimic sequence length 1
        support_g = self.support_encoder(support_vec.unsqueeze(1))
        
        # 2. Encode Query (FIXED: Added this step!)
        # Previously we passed raw query_vec. Now we project it same as support.
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))
        
        # 3. Mean Pooling (FIXED: Added this step!)
        # Stabilizes the support representation by averaging
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        
        # 4. Squeeze back for LSTM
        support_g = support_g.squeeze(1)
        query_encoded = query_encoded.squeeze(1)

        # 5. LSTM Query Refinement
        query_g = self.query_encoder(support_g, query_encoded)
        
        # Calculate Similarity
        matching_scores = torch.sum(query_g * support_g, dim=1)
        
        return matching_scores

    def run_baseline_logic(self, query, support):
        """Helper for baseline logic"""
        s = self.dropout(self.symbol_emb(support)).view(-1, 2*self.embed_dim)
        q = self.dropout(self.symbol_emb(query)).view(-1, 2*self.embed_dim)
        
        # Standard Baseline Flow
        s_g = self.support_encoder(s.unsqueeze(1))
        q_encoded = self.support_encoder(q.unsqueeze(1))
        
        s_g = torch.mean(s_g, dim=0, keepdim=True)
        
        s_g = s_g.squeeze(1)
        q_encoded = q_encoded.squeeze(1)
        
        q_f = self.query_encoder(s_g, q_encoded)
        return torch.sum(q_f * s_g, dim=1)
