import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder

class EmbedMatcher(nn.Module):
    """
    Robust 1-Hop Matcher with Soft Distance Gating.
    Fixes: Gradient flow, Self-Loop collapse, and Gate saturation.
    """

    def __init__(
        self,
        embed_dim,
        num_symbols,
        use_pretrain=True,
        embed=None,
        dropout=0.2,
        batch_size=64,
        process_steps=4,
        finetune=False,
        aggregate='max',
        gate_temp=1.0  # Temperature for the distance softmax
    ):
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.dropout = nn.Dropout(dropout)
        self.gate_temp = gate_temp

        self.symbol_emb = nn.Embedding(
            num_symbols + 1,
            embed_dim,
            padding_idx=self.pad_idx
        )

        # --- Improved Gating Components ---
        # We use a Residual Gate: Output = Self + Alpha * Neighbors
        # This prevents the "Dying Gate" problem.
        
        # Transformation for the Neighbor (Relation + Entity)
        self.gcn_w = nn.Linear(embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))
        
        # Gate Control: Decides how much neighbor info to mix in
        self.gate_w = nn.Linear(embed_dim * 2, 1) 
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.gate_w.weight)
        # Initialize bias to 1.0 to start with the gate "open"
        init.constant_(self.gate_b, 1.0) 

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
        Encodes 1-Hop neighbors using Differentiable Distance Weighting.
        """
        # connections: [Batch, Max_Neighbors, 2]
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]

        # 1. Embeddings
        rel_emb = self.symbol_emb(relations)     # [Batch, K, Dim]
        ent_emb = self.symbol_emb(entities)      # [Batch, K, Dim]
        self_emb = self.symbol_emb(entity_self_ids).unsqueeze(1) # [Batch, 1, Dim]
        
        # Apply dropout *after* retrieval
        rel_emb = self.dropout(rel_emb)
        ent_emb = self.dropout(ent_emb)
        self_emb = self.dropout(self_emb)

        # 2. Calculate Distance-Based Weights (Soft Attention)
        # We use Dot Product as a proxy for similarity (efficient for high dim)
        # Normalize to ensure stability
        norm_self = F.normalize(self_emb, p=2, dim=-1)
        norm_neigh = F.normalize(ent_emb, p=2, dim=-1)
        
        # Similarity: [Batch, 1, K] -> [Batch, K]
        # High score = Close distance
        raw_sim = torch.bmm(norm_self, norm_neigh.transpose(1, 2)).squeeze(1)
        
        # Mask Padding (Critical Step)
        mask = (relations != self.pad_idx).float()
        # Set padding scores to -infinity so Softmax ignores them
        raw_sim = raw_sim * mask + (1 - mask) * -1e9

        # Softmax turns distances into differentiable weights
        # We divide by gate_temp to sharpen/smooth the distribution
        neighbor_weights = F.softmax(raw_sim / self.gate_temp, dim=-1).unsqueeze(-1)

        # 3. Combine Relation + Entity
        # Instead of Concat, we use Element-wise Product (ComplEx style interaction)
        # followed by a Linear projection. This is stronger for KG.
        neighbor_msg = rel_emb * ent_emb 
        neighbor_msg = self.gcn_w(neighbor_msg) + self.gcn_b
        neighbor_msg = F.leaky_relu(neighbor_msg)

        # 4. Weighted Aggregation (Distance-Based)
        # [Batch, K, Dim] * [Batch, K, 1] -> Sum -> [Batch, Dim]
        weighted_agg = torch.sum(neighbor_msg * neighbor_weights, dim=1)

        # 5. Residual Gating
        # We compute a gate based on Self AND the Aggregated Context
        gate_input = torch.cat([self_emb.squeeze(1), weighted_agg], dim=-1)
        gate_val = torch.sigmoid(self.gate_w(gate_input) + self.gate_b)
        
        # Final = Self + Gate * Neighbors
        # This guarantees that 'Self' info is preserved, preventing collapse.
        output = self_emb.squeeze(1) + (gate_val * weighted_agg)

        return torch.tanh(output)

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        # Unpack meta (Compatible with your loader)
        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta

        # --- ENCODE QUERY ---
        # Pass self_ids to neighbor_encoder for distance calculation
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        
        # Concatenate Head+Tail vectors
        query_vec = torch.cat((q_left, q_right), dim=-1)

        # --- ENCODE SUPPORT ---
        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)

        # --- MATCHING NETWORK ---
        # (Standard matching logic from original code)
        support_g = self.support_encoder(support_vec.unsqueeze(1))
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))

        support_g = torch.mean(support_g, dim=0, keepdim=True)
        support_g = support_g.squeeze(1)
        query_encoded = query_encoded.squeeze(1)

        query_g = self.query_encoder(support_g, query_encoded)

        scores = torch.sum(query_g * support_g, dim=1)
        return scores
