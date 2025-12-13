import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder

class EmbedMatcher(nn.Module):
    """
    Relation-Aware Gating + Distance Filtering for Complex Embeddings
    Uses joint cosine similarity over (real, imag) parts and applies gating only to relevant relations.
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
        self.gate_temp = nn.Parameter(torch.tensor(1.0))  # learnable temperature

        # Embedding layer
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # Neighbor projection
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))

        # Relation-aware gating (one gate per relation)
        self.gate_w = nn.Embedding(num_symbols, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        # Initialization
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.constant_(self.gate_b, 0)
        init.normal_(self.gate_w.weight, mean=1.0, std=0.05)

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
        Neighbor encoding with gating and joint cosine similarity for complex embeddings.
        """
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        batch_size, max_k = entities.shape

        # Embeddings
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))
        self_embeds = self.symbol_emb(entity_self_ids).unsqueeze(1)

        # Concatenate relation + entity embeddings
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        projected = F.leaky_relu(self.gcn_w(concat_embeds) + self.gcn_b)

        # Mask padding
        is_pad = (relations == self.pad_idx).float().unsqueeze(-1)
        projected = projected * (1.0 - is_pad)

        # Joint Cosine Similarity for complex embeddings
        d = projected.shape[-1] // 2
        real_self = self_embeds[..., :d]
        imag_self = self_embeds[..., d:]
        real_neigh = projected[..., :d]
        imag_neigh = projected[..., d:]

        numerator = torch.sum(real_self * real_neigh, dim=-1) + torch.sum(imag_self * imag_neigh, dim=-1)
        norm_self = torch.sqrt(torch.sum(real_self**2 + imag_self**2, dim=-1) + 1e-8)
        norm_neigh = torch.sqrt(torch.sum(real_neigh**2 + imag_neigh**2, dim=-1) + 1e-8)
        cos_joint = numerator / (norm_self * norm_neigh + 1e-8)

        # Take top-k neighbors
        topk_vals, topk_idx = torch.topk(cos_joint, k=min(self.k_neighbors, max_k), dim=1)
        neighbor_agg = torch.gather(projected, 1, topk_idx.unsqueeze(-1).expand(-1, -1, projected.size(-1)))
        neighbor_agg = neighbor_agg.mean(dim=1)

        # Relation-aware gating for relevant neighbors
        neighbor_rel_ids = relations.clone()
        neighbor_rel_ids[relations == self.pad_idx] = 0
        gate_vals = self.gate_w(neighbor_rel_ids).squeeze(-1)
        gate_vals = torch.sigmoid(gate_vals.mean(dim=1, keepdim=True) / self.gate_temp)
        gate_vals = torch.where(num_neighbors.unsqueeze(-1) > 0, gate_vals, torch.ones_like(gate_vals))

        # Residual connection
        final_vec = self_embeds.squeeze(1) + gate_vals * neighbor_agg
        return torch.tanh(final_vec)

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

        support_g = torch.mean(support_g, dim=0, keepdim=True)
        support_g = support_g.squeeze(1)
        query_encoded = query_encoded.squeeze(1)

        query_g = self.query_encoder(support_g, query_encoded)

        # Use joint cosine similarity for final score as well
        d = support_g.shape[-1] // 2
        real_q, imag_q = query_g[..., :d], query_g[..., d:]
        real_s, imag_s = support_g[..., :d], support_g[..., d:]
        numerator = torch.sum(real_q * real_s, dim=-1) + torch.sum(imag_q * imag_s, dim=-1)
        denom = torch.sqrt(torch.sum(real_q**2 + imag_q**2, dim=-1) + 1e-8) * torch.sqrt(torch.sum(real_s**2 + imag_s**2, dim=-1) + 1e-8)
        scores = numerator / denom

        return scores
