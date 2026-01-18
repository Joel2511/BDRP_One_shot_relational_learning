import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import faiss
import numpy as np
from modules import *

class EmbedMatcher(nn.Module):
    """
    k-NN + MEAN Aggregation for Medical Knowledge Graphs
    Optimized for 100D Structure-Preserving Embeddings
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, 
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, 
                 aggregate='max', knn_k=32, knn_path=None, knn_alpha=0.5):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols
        self.knn_k = knn_k
        self.knn_alpha = knn_alpha  # structural + knn_mean interpolation

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain and embed is not None:
            logging.info('LOADING KB EMBEDDINGS INTO MATCHER')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING (Non-Trainable)')
                self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)
        
        # k-NN Components
        self.knn_index = None
        self.knn_neighbors = None 
        if knn_path is not None:
            self.load_knn_index(knn_path)

    def joint_cosine_similarity(self, emb1, emb2):
        """Standard cosine similarity for 100D real+imaginary concatenated vectors"""
        return F.cosine_similarity(emb1, emb2, dim=-1)

    def knn_neighbor_encoder(self, entity_ids):
        """Encodes semantic neighbors identified via FAISS"""
        if self.knn_neighbors is None:
            return None
            
        # Move neighbors to same device as model
        device = self.symbol_emb.weight.device
        knn_idx = self.knn_neighbors[entity_ids].to(device) # [B, k]
        
        # In medical KG, we treat k-NN neighbors as entities sharing a virtual relation
        knn_rels = torch.zeros_like(knn_idx).to(device)
        knn_ents = knn_idx
        
        rel_embeds = self.dropout(self.symbol_emb(knn_rels))
        ent_embeds = self.dropout(self.symbol_emb(knn_ents))
        
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)
        
        # MEAN POOLING for semantic neighbors
        knn_mean = torch.mean(out, dim=1).tanh()
        
        return knn_mean

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
        """
        MEAN Aggregation: Essential for capturing context in dense hubs (Avg Deg 63)
        """
        num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))
        
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)
        
        # Aggregation: Sum / Count (Mean Pooling)
        # Research shows Max-Pooling is too reductive for deep medical hierarchies
        out = torch.sum(out, dim=1) / num_neighbors
        structural_repr = out.tanh()

        # Hybrid Enrichment (Semantic k-NN)
        knn_mean = None
        if entity_ids is not None and self.knn_neighbors is not None:
            knn_mean = self.knn_neighbor_encoder(entity_ids)

        if knn_mean is not None:
            # Interpolation logic: Balance between structure and semantics
            final = self.knn_alpha * structural_repr + (1 - self.knn_alpha) * knn_mean
        else:
            final = structural_repr
            
        return final

    def forward(self, query, support, query_meta=None, support_meta=None):
        """
        query: (batch_size, 2) -> [head, tail]
        support: (few, 2) -> [head, tail]
        """
        if query_meta is None or support_meta is None:
            # Fallback for non-graph runs
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))

        # Handle metadata unpacking (supports 6-tuple or 4-tuple)
        if len(query_meta) == 6:
            q_l_conn, _, q_l_deg, q_r_conn, _, q_r_deg = query_meta
            s_l_conn, _, s_l_deg, s_r_conn, _, s_r_deg = support_meta
        else:
            q_l_conn, q_l_deg, q_r_conn, q_r_deg = query_meta
            s_l_conn, s_l_deg, s_r_conn, s_r_deg = support_meta

        # Extract IDs for k-NN enrichment
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]

        # Process Left (Head) and Right (Tail) Contexts
        query_left = self.neighbor_encoder(q_l_conn, q_l_deg, q_h_ids)
        query_right = self.neighbor_encoder(q_r_conn, q_r_deg, q_t_ids)
        support_left = self.neighbor_encoder(s_l_conn, s_l_deg, s_h_ids)
        support_right = self.neighbor_encoder(s_r_conn, s_r_deg, s_t_ids)

        # Combine head and tail representations
        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)

        # Matching Network (G-Matching Logic)
        support_g = self.support_encoder(support_neighbor.unsqueeze(0))
        support_g = torch.mean(support_g, dim=1) # Aggregate support set
        
        query_g = self.support_encoder(query_neighbor.unsqueeze(1))
        query_g = query_g.squeeze(1)
        
        # Query Encoding (Refining query relative to support)
        query_f = self.query_encoder(support_g, query_g)
        
        # Dot product for final matching score
        matching_scores = torch.matmul(query_f, support_g.squeeze(0).t()).squeeze(-1)
        return matching_scores

    def load_knn_index(self, knn_path):
        """Safe load of precomputed FAISS indices"""
        try:
            self.knn_neighbors = torch.load(knn_path)
            self.knn_k = self.knn_neighbors.shape[1]
            logging.info(f'LOADED k-NN: {self.knn_neighbors.shape}')
        except Exception as e:
            logging.error(f"Failed to load k-NN index: {e}")

if __name__ == '__main__':
    # Test for 100D Structure (50 real + 50 imag)
    query = torch.ones(64, 2).long() * 100
    support = torch.ones(1, 2).long() 
    matcher = EmbedMatcher(100, 40000, use_pretrain=False, knn_k=32)
    dummy_meta = [torch.zeros(64, 50, 2).long(), torch.ones(64)] * 2
    print(f"Output Score Shape: {matcher(query, support, dummy_meta, dummy_meta).size()}")
