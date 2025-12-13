import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings - MAX-POOL VERSION
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.dropout = nn.Dropout(dropout)  # Configurable dropout

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain and embed is not None:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        MAX-POOL AGGREGATION (SOTA for NELL-One)
        '''
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        
        # Embeddings
        rel_embeds = self.symbol_emb(relations)  # [B, K, D]
        ent_embeds = self.symbol_emb(entities)   # [B, K, D]
        
        # Mask padding neighbors
        rel_pad_mask = (relations == self.pad_idx)
        ent_pad_mask = (entities == self.pad_idx)
        pad_mask = rel_pad_mask | ent_pad_mask
        
        # GCN projection
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)  # [B, K, 2D]
        projected = self.gcn_w(concat_embeds) + self.gcn_b           # [B, K, D]
        projected = F.leaky_relu(projected, 0.1)                      # Stable activation
        
        # Apply dropout BEFORE masking
        projected = self.dropout(projected)
        
        # MAX-POOL (ignores padding via -inf masking)
        projected = projected.masked_fill(pad_mask.unsqueeze(-1), float('-inf'))
        neighbor_agg = projected.max(dim=1)[0]  # [B, D] - KEY CHANGE
        
        return torch.tanh(neighbor_agg)

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        if query_meta is None or support_meta is None:
            # Fallback for no neighbors
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))

        # Original 4-tuple unpacking
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        # Neighbor encoding
        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)

        # Matching network (UNCHANGED)
        support_g = self.support_encoder(support_neighbor.unsqueeze(0))  # [1, 1, 2D]
        support_g = torch.mean(support_g, dim=1)                         # [1, 2D]
        query_g = self.support_encoder(query_neighbor.unsqueeze(1))      # [B, 1, 2D]
        query_g = query_g.squeeze(1)                                     # [B, 2D]

        query_f = self.query_encoder(support_g, query_g)                 # [B, 2D]

        matching_scores = torch.matmul(query_f, support_g.squeeze(0).t()).squeeze(-1)
        return matching_scores

    def forward_(self, query_meta, support_meta):
        """Legacy method - unchanged"""
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1)
        support = torch.cat((support_left, support_right), dim=-1)
        
        support_expand = support.expand_as(query)
        # Note: siamese not defined - this is legacy
        return torch.zeros(query.size(0))

if __name__ == '__main__':
    query = torch.ones(64,2).long() * 100
    support = torch.ones(40,2).long()
    matcher = EmbedMatcher(200, 40000, use_pretrain=False, dropout=0.1)
    
    # Dummy meta for testing
    dummy_meta = [torch.zeros(64,50,2).long(), torch.ones(64), 
                  torch.zeros(64,50,2).long(), torch.ones(64)]
    print(matcher(query, support, dummy_meta, dummy_meta).size())
