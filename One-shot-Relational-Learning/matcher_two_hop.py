import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

# Assuming SupportEncoder and QueryEncoder are defined in modules.py
# If not, they need to be added here or imported correctly.
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

        # --- GANA-Specific Components ---
        
        # 1. Neighbor Projection (The GCN part)
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
        init.constant_(self.attn_linear.bias, 0)
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
        Encodes neighbors using Attention and Gating.
        Formula: Final = Gate * Neighbor_Agg + (1 - Gate) * Self_Emb
        
        Args:
            connections: (Batch, Max_Neighbors, 2) - [Relation_ID, Entity_ID]
            num_neighbors: (Batch,) - Number of valid neighbors (unused if masking is simple)
            entity_self_ids: (Batch,) - Symbol IDs of the center entities
        """
        # 1. Retrieve Embeddings
        # connections[:, :, 0] is Relation, [:, :, 1] is Tail Entity
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]

        rel_emb = self.dropout(self.symbol_emb(relations)) # (Batch, Neigh, Dim)
        ent_emb = self.dropout(self.symbol_emb(entities)) # (Batch, Neigh, Dim)
        
        # Get "Self" embedding for the Residual Connection
        # Shape: (Batch, 1, Dim)
        self_emb = self.dropout(self.symbol_emb(entity_self_ids)).unsqueeze(1) 

        # 2. GCN Projection (Relation + Entity -> Feature)
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      # (Batch, Neigh, 2*Dim)
        projected = self.gcn_w(concat) + self.gcn_b         # (Batch, Neigh, Dim)
        projected = F.leaky_relu(projected)

        # 3. Attention Mechanism
        # Calculate attention scores
        attn_scores = self.attn_linear(projected)           # (Batch, Neigh, 1)
        
        # Apply Softmax to get weights (Values sum to 1 across neighbors)
        # Note: For variable neighbors, you might mask here, but standard GMatching often relies on padding
        attn_weights = F.softmax(attn_scores, dim=1)        # (Batch, Neigh, 1)
        
        # Weighted Sum of neighbors
        neighbor_agg = torch.sum(projected * attn_weights, dim=1) # (Batch, Dim)

        # 4. Gating (The "Noise Filter")
        # Compute gate value between 0 and 1
        gate_val = torch.sigmoid(self.gate_w(neighbor_agg) + self.gate_b) # (Batch, 1)
        
        # 5. Residual Connection with Gating
        # If Gate is 1, we trust the Neighbors.
        # If Gate is 0, we ignore Neighbors and use Self.
        self_vec = self_emb.squeeze(1)
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_vec)

        return final_vec

    def forward(self, query, support, query_meta=None, support_meta=None):
        """
        Forward pass with Self-Gating and Dimension Corrections.
        """
        # Fallback for Baseline mode (if no meta provided)
        if (query_meta is None) or (support_meta is None):
            return self.run_baseline_logic(query, support)

        # Extract Entity IDs for Self-Gating
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        # Unpack Meta (Graph Context)
        (q_l1, q_l2, q_deg_l, q_r1, q_r2, q_deg_r) = query_meta
        (s_l1, s_l2, s_deg_l, s_r1, s_r2, s_deg_r) = support_meta
        
        # --- 1. Encode Query Entity Pairs ---
        # Hop 1
        q_h_1 = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_t_1 = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        # Hop 2
        q_h_2 = self.neighbor_encoder(q_l2, None, q_h_ids)
        q_t_2 = self.neighbor_encoder(q_r2, None, q_t_ids)

        # Fuse Hops (Average)
        q_left = (q_h_1 + q_h_2) / 2.0
        q_right = (q_t_1 + q_t_2) / 2.0

        # Concatenate to form Pair Representation
        query_vec = torch.cat((q_left, q_right), dim=-1) # Shape: [Batch, 2*Dim]

        # --- 2. Encode Support Entity Pairs ---
        # Hop 1
        s_h_1 = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_t_1 = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        # Hop 2
        s_h_2 = self.neighbor_encoder(s_l2, None, s_h_ids)
        s_t_2 = self.neighbor_encoder(s_r2, None, s_t_ids)

        # Fuse Hops
        s_left = (s_h_1 + s_h_2) / 2.0
        s_right = (s_t_1 + s_t_2) / 2.0

        # Concatenate
        support_vec = torch.cat((s_left, s_right), dim=-1) # Shape: [Batch, 2*Dim]
        
        # --- 3. Matching Network (FIXED) ---
        # Support Encoder is flexible (Linear layers), so 2D input [Batch, Dim] works fine.
        # We DO NOT unsqueeze here, keeping it 2D.
        support_g = self.support_encoder(support_vec) 
        
        # Query Encoder uses LSTMCell. It STRICTLY requires 2D input [Batch, Dim].
        # We pass query_vec (2D) and support_g (2D) directly.
        query_g = self.query_encoder(support_g, query_vec)
        
        # Calculate Similarity (Dot Product)
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
