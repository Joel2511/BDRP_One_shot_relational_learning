import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings
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

        # --- Gating parameters added later ---
        self.gate_linear = nn.Linear(2*self.embed_dim, 1)  # Linear layer for gating
        init.xavier_normal_(self.gate_linear.weight)
        init.constant_(self.gate_linear.bias, 0)

        self.dropout = nn.Dropout(0.5)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain:
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
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)

        # --- Gating mechanism applied here ---
        gate_values = torch.sigmoid(self.gate_linear(concat_embeds))  # (batch, 200, 1)
        gated_embeds = gate_values * concat_embeds  # Element-wise multiplication

        out = self.gcn_w(gated_embeds) + self.gcn_b

        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    # encode_neighbors unchanged; uses gated neighbor_encoder implicitly
    def encode_neighbors(self, neighbors_1hop, neighbors_2hop, degrees):
        enc_1hop = self.neighbor_encoder(neighbors_1hop, degrees)
        enc_2hop = self.neighbor_encoder(neighbors_2hop, degrees)
        combined = enc_1hop + enc_2hop  # sum aggregation; can consider concat or weighted sum later
        return combined

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''

        if query_meta is None or support_meta is None:
            # fallback if no meta info provided
            support = self.dropout(self.symbol_emb(support)).view(-1, 2*self.embed_dim)
            query = self.dropout(self.symbol_emb(query)).view(-1, 2*self.embed_dim)
            support = support.unsqueeze(0)
            support_g = self.support_encoder(support).squeeze(0)
            query_f = self.query_encoder(support_g, query)
            matching_scores = torch.matmul(query_f, support_g.t()).squeeze()
            return matching_scores

        # Unpack 1-hop and 2-hop neighbors and degrees from meta
        query_left_1hop, query_left_2hop, query_left_degrees, query_right_1hop, query_right_2hop, query_right_degrees = query_meta
        support_left_1hop, support_left_2hop, support_left_degrees, support_right_1hop, support_right_2hop, support_right_degrees = support_meta

        query_left = self.encode_neighbors(query_left_1hop, query_left_2hop, query_left_degrees)
        query_right = self.encode_neighbors(query_right_1hop, query_right_2hop, query_right_degrees)

        support_left = self.encode_neighbors(support_left_1hop, support_left_2hop, support_left_degrees)
        support_right = self.encode_neighbors(support_right_1hop, support_right_2hop, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1) # (batch, 2*embed_dim)
        support_neighbor = torch.cat((support_left, support_right), dim=-1) # (few, 2*embed_dim)

        support_g = self.support_encoder(support_neighbor) 
        query_g = self.query_encoder(support_g, query_neighbor)

        support_g = torch.mean(support_g, dim=0, keepdim=True)

        matching_scores = torch.matmul(query_g, support_g.t()).squeeze()

        return matching_scores

    def forward_(self, query_meta, support_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1) # tanh
        support = torch.cat((support_left, support_right), dim=-1) # tanh

        support_expand = support.expand_as(query)

        distances = F.sigmoid(self.siamese(torch.abs(support_expand - query))).squeeze()
        return distances

if __name__ == '__main__':

    query = Variable(torch.ones(64,2).long()) * 100
    support = Variable(torch.ones(40,2).long())
    matcher = EmbedMatcher(40, 200, use_pretrain=False)
    print(matcher(query, support).size())
