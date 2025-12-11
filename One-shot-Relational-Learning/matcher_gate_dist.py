import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import SupportEncoder, QueryEncoder 

class EmbedMatcher(nn.Module):
    """
    1-Hop Gating + Distance Filtering.
    Replaces Attention weights with fixed weights derived from Cosine Similarity
    to ensure stability in the One-Shot setting.
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None,
                 dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', 
                 gate_temp=1.0, k_neighbors=10): 
        super(EmbedMatcher, self).__init__()

        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.num_symbols = num_symbols
        self.aggregate = aggregate
        self.dropout = nn.Dropout(dropout)
        self.gate_temp = gate_temp
        self.k_neighbors = k_neighbors 

        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)

        # --- Gating Components (Projection/Gate remains) ---
        self.gcn_w = nn.Linear(2 * embed_dim, embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(embed_dim))
        
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

    def neighbor_encoder(self, connections, num_neighbors, entity_self_ids, query_relation=None):
        """
        Encodes 1-Hop neighbors using Distance Filtering (Top-K) + Gating.
        
        Args:
            connections: [Batch, K_max, 2] neighbor connections (relation_id, entity_id)
            num_neighbors: number of neighbors per entity
            entity_self_ids: [Batch] center entity IDs
            query_relation: [Batch] (optional) query relation IDs for relation-aware filtering
        """
        relations = connections[:, :, 0]
        entities = connections[:, :, 1]
        
        rel_emb = self.dropout(self.symbol_emb(relations)) # [Batch, K_max, Dim]
        ent_emb = self.dropout(self.symbol_emb(entities))  # [Batch, K_max, Dim]
        self_emb = self.dropout(self.symbol_emb(entity_self_ids)).unsqueeze(1) # [Batch, 1, Dim]
        
        batch_size, max_k, dim = ent_emb.shape
        
        # --- 1. Distance Calculation (Semantic Proximity) ---
        # Normalize vectors for Cosine Similarity
        norm_self = F.normalize(self_emb, p=2, dim=-1, eps=1e-8) # [Batch, 1, Dim]
        norm_neigh = F.normalize(ent_emb, p=2, dim=-1, eps=1e-8) # [Batch, K_max, Dim]
        
        # Entity Cosine Similarity: [Batch, K_max]
        entity_similarity = torch.bmm(norm_self, norm_neigh.transpose(1, 2)).squeeze(1)
        
        # Optional: Relation-aware similarity
        if query_relation is not None:
            query_rel_emb = self.dropout(self.symbol_emb(query_relation)).unsqueeze(1) # [Batch, 1, Dim]
            norm_query_rel = F.normalize(query_rel_emb, p=2, dim=-1, eps=1e-8)
            norm_rel = F.normalize(rel_emb, p=2, dim=-1, eps=1e-8)
            rel_similarity = torch.bmm(norm_query_rel, norm_rel.transpose(1, 2)).squeeze(1)
            
            # Combine entity and relation similarity (tune these weights)
            similarity_scores = 0.7 * entity_similarity + 0.3 * rel_similarity
        else:
            similarity_scores = entity_similarity
        
        # --- 2. Top-K Filtering (Graph Structure Learning) ---
        # Mask out padding indices (where relation ID is PAD)
        is_pad = (relations == self.pad_idx)
        similarity_scores = similarity_scores.masked_fill(is_pad, float('-inf'))
        
        # Get top K indices
        k_to_select = min(self.k_neighbors, max_k)
        
        # Use torch.topk to get the K highest scores
        k_scores, k_indices = torch.topk(similarity_scores, k=k_to_select, dim=-1)
        
        # Create selection mask
        current_device = self.symbol_emb.weight.device
        k_mask = torch.zeros((batch_size, max_k), device=current_device)
        k_mask.scatter_(1, k_indices, 1) # Set selected indices to 1
        
        # Check if selected neighbors are padding
        is_pad_in_topk = is_pad.gather(1, k_indices) # [Batch, k_to_select]
        
        # Expand masks for broadcasting
        k_mask = k_mask.unsqueeze(-1) # [Batch, K_max, 1]
        
        # --- 3. GCN Projection & Aggregation ---
        concat = torch.cat((rel_emb, ent_emb), dim=-1)      
        projected = self.gcn_w(concat) + self.gcn_b         
        projected = F.leaky_relu(projected)
        
        # Apply the mask to projected features
        filtered_projected = projected * k_mask.float()
        
        # CRITICAL FIX: Count actual valid (non-padding) neighbors in top-k
        valid_neighbor_mask = k_mask.squeeze(-1) * (~is_pad).float() # [Batch, K_max]
        actual_count = torch.sum(valid_neighbor_mask, dim=1, keepdim=True) # [Batch, 1]
        
        # Take MEAN of only valid neighbors
        neighbor_agg = torch.sum(filtered_projected, dim=1) / (actual_count + 1e-9) # [Batch, Dim]
        
        # 4. Gating (Softened by Temperature)
        gate_input = self.gate_w(neighbor_agg) + self.gate_b
        gate_val = torch.sigmoid(gate_input / self.gate_temp) 
        
        # 5. Residual Connection
        final_vec = (gate_val * neighbor_agg) + ((1.0 - gate_val) * self_emb.squeeze(1))
        
        return torch.tanh(final_vec)

    def forward(self, query, support, query_meta=None, support_meta=None):
        q_h_ids, q_t_ids = query[:, 0], query[:, 1]
        s_h_ids, s_t_ids = support[:, 0], support[:, 1]
        
        # Extract relation IDs for relation-aware filtering
        q_relation = query[:, 2] if query.shape[1] > 2 else None
        s_relation = support[:, 2] if support.shape[1] > 2 else None
        
        (q_l1, _, q_deg_l, q_r1, _, q_deg_r) = query_meta
        (s_l1, _, s_deg_l, s_r1, _, s_deg_r) = support_meta
        
        # --- ENCODE QUERY ---
        # Note: We are using a 1-Hop Gated Matcher here.
        # For head entity: use query relation for left neighbors
        q_left = self.neighbor_encoder(q_l1, q_deg_l, q_h_ids, query_relation=q_relation)
        # For tail entity: use query relation for right neighbors  
        q_right = self.neighbor_encoder(q_r1, q_deg_r, q_t_ids, query_relation=q_relation)
        query_vec = torch.cat((q_left, q_right), dim=-1)
        
        # --- ENCODE SUPPORT ---
        s_left = self.neighbor_encoder(s_l1, s_deg_l, s_h_ids, query_relation=s_relation)
        s_right = self.neighbor_encoder(s_r1, s_deg_r, s_t_ids, query_relation=s_relation)
        support_vec = torch.cat((s_left, s_right), dim=-1)
        
        # --- MATCHING NETWORK (Standard Baseline Pipeline) ---
        support_g = self.support_encoder(support_vec.unsqueeze(1)) 
        query_encoded = self.support_encoder(query_vec.unsqueeze(1))
        
        # Mean Pooling of support set
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        
        support_g = support_g.squeeze(1) 
        query_encoded = query_encoded.squeeze(1) 
        
        # LSTM Query Refinement
        query_g = self.query_encoder(support_g, query_encoded)
        
        # Scoring
        scores = torch.sum(query_g * support_g, dim=1)
        return scores
