import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder

class EmbedMatcher(nn.Module):
    """
    Relation-Aware Gating + Query-Attention Neighbor Aggregation
    Ensures baseline neighbor aggregation behavior with query-adaptive attention weights.
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

        # Embedding layer
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # Neighbor projection
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # Relation-aware gating (optional bias on attention scores)
        self.gate_w = nn.Embedding(num_symbols, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.constant_(self.gate_b, 0)
        init.normal_(self.gate_w.weight, mean=0.0, std=0.05)  # small bias around 0

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
        Neighbor encoding with query-dependent attention and optional relation bias.
        """
        relations = connections[:, :, 0]           # (batch, max_k)
        entities = connections[:, :, 1]            # (batch, max_k)
        batch_size, max_k = entities.shape

        # Embeddings
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))
        self_embeds = self.symbol_emb(entity_self_ids).unsqueeze(1)  # (batch,1,embed_dim)

        # Concatenate relation + entity embeddings
        neighbor_vecs = torch.cat((rel_embeds, ent_embeds), dim=-1)  # (batch,max_k,2*embed_dim)
        projected = F.leaky_relu(self.gcn_w(neighbor_vecs) + self.gcn_b)  # (batch,max_k,embed_dim)

        # Mask padding
        is_pad = (relations == self.pad_idx).float().unsqueeze(-1)
        projected = projected * (1.0 - is_pad)

        # Query-dependent attention
        query_vec = self_embeds  # (batch,1,embed_dim)
        attn_scores = torch.bmm(projected, query_vec.transpose(1,2)).squeeze(-1)  # (batch,max_k)
        # Add relation bias
        gate_vals = self.gate_w(relations).squeeze(-1) + self.gate_b  # (batch,max_k)
        attn_scores = attn_scores + gate_vals

        # Mask padding in attention
        attn_scores = attn_scores.masked_fill(relations == self.pad_idx, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch,max_k,1)

        # Weighted sum
        neighbor_agg = (projected * attn_weights).sum(dim=1)  # (batch,embed_dim)

        # Residual + tanh
        final_vec = self_embeds.squeeze(1) + neighbor_agg
        return torch.tanh(final_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta

        # Neighbor aggregation with attention
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids)
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids)
        query_vec = torch.cat((q_left, q_right), dim=-1)

        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids)
        support_vec = torch.cat((s_left, s_right), dim=-1)

        # Encode via SupportEncoder
        support_g = self.support_encoder(support_vec.unsqueeze(1))
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))

        support_g = torch.mean(support_g, dim=0, keepdim=True).squeeze(1)
        query_encoded = query_encoded.squeeze(1)

        # Query encoder
        query_g = self.query_encoder(support_g, query_encoded)

        # Cosine similarity (or dot product)
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
