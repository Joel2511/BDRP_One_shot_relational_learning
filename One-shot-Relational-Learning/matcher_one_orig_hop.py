import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder

class EmbedMatcher(nn.Module):
    """
    Robust Gated Mean-Pooling Matcher.
    Removes the risky 'Distance Gating' to ensure all structural neighbors 
    are considered, regardless of vector similarity.
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
        gate_temp=1.0 
    ):
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.dropout = nn.Dropout(dropout)

        self.symbol_emb = nn.Embedding(
            num_symbols + 1,
            embed_dim,
            padding_idx=self.pad_idx
        )

        # --- Simplified Interaction & Gating ---
        # Projects neighbor information (Relation * Entity)
        self.proj_w = nn.Linear(embed_dim, embed_dim)
        self.proj_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # The Gate: Decides [Self vs Neighbors]
        self.gate_w = nn.Linear(embed_dim * 2, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        init.xavier_normal_(self.proj_w.weight)
        init.constant_(self.proj_b, 0)
        init.xavier_normal_(self.gate_w.weight)
        init.constant_(self.gate_b, 0.0) # Start neutral

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
        # connections: [Batch, K, 2]
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]

        # 1. Embeddings
        rel_emb = self.symbol_emb(relations)   # [Batch, K, Dim]
        ent_emb = self.symbol_emb(entities)    # [Batch, K, Dim]
        self_emb = self.symbol_emb(entity_self_ids).unsqueeze(1) # [Batch, 1, Dim]

        rel_emb = self.dropout(rel_emb)
        ent_emb = self.dropout(ent_emb)
        self_emb = self.dropout(self_emb)

        # 2. Neighbor Interaction (Element-wise product for ComplEx)
        # We project this to mix the features
        neighbor_vec = rel_emb * ent_emb
        neighbor_vec = self.proj_w(neighbor_vec) + self.proj_b
        neighbor_vec = F.leaky_relu(neighbor_vec)

        # 3. Masking Padding
        mask = (relations != self.pad_idx).float().unsqueeze(-1) # [Batch, K, 1]
        neighbor_vec = neighbor_vec * mask

        # 4. Mean Aggregation (Safe & Robust)
        # Sum of neighbors / (Number of real neighbors + epsilon)
        # This is much safer than Distance Softmax for 1-hop
        real_neighbor_counts = torch.sum(mask, dim=1) # [Batch, 1]
        neighbor_agg = torch.sum(neighbor_vec, dim=1) / (real_neighbor_counts + 1e-9)

        # 5. Gating (The "One-Shot" Logic)
        # Gate looks at Self AND the context to decide importance
        gate_input = torch.cat([self_emb.squeeze(1), neighbor_agg], dim=-1)
        gate_val = torch.sigmoid(self.gate_w(gate_input) + self.gate_b)

        # Final = Self + Gate * Context
        output = self_emb.squeeze(1) + (gate_val * neighbor_agg)
        
        return torch.tanh(output)

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

        support_g = torch.mean(support_g, dim=0, keepdim=True).squeeze(1)
        query_encoded = query_encoded.squeeze(1)

        query_g = self.query_encoder(support_g, query_encoded)
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
