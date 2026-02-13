import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    def __init__(self, embed_dim, num_symbols, semantic_dim=0, use_pretrain=True, embed=None, dropout=0.2,
                 batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.semantic_dim = semantic_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2*self.embed_dim + self.semantic_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(0.5)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain and embed is not None:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors, semantic=None):
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:,:,0]
        entities = connections[:,:,1]

        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        if semantic is not None:
            semantic_embeds = semantic[entities]
            concat_embeds = torch.cat((rel_embeds, ent_embeds, semantic_embeds), dim=-1)
        else:
            concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)

        out = self.gcn_w(concat_embeds) + self.gcn_b
        out = torch.sum(out, dim=1) / num_neighbors
        return out.tanh()

    def forward(self, query, support, query_meta=None, support_meta=None, query_sem=None, support_sem=None):
        if query_meta is not None:
            q_left_conn, q_left_deg, q_right_conn, q_right_deg = query_meta
            s_left_conn, s_left_deg, s_right_conn, s_right_deg = support_meta

            query_left = self.neighbor_encoder(q_left_conn, q_left_deg, semantic=query_sem)
            query_right = self.neighbor_encoder(q_right_conn, q_right_deg, semantic=query_sem)
            support_left = self.neighbor_encoder(s_left_conn, s_left_deg, semantic=support_sem)
            support_right = self.neighbor_encoder(s_right_conn, s_right_deg, semantic=support_sem)
            query_neighbor = torch.cat((query_left, query_right), dim=-1)
            support_neighbor = torch.cat((support_left, support_right), dim=-1)
        else:
            query_neighbor = query.float()
            support_neighbor = support.float()

        support_g = self.support_encoder(support_neighbor)
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        query_f = self.query_encoder(support_g, query_neighbor)

        return torch.matmul(query_f, support_g.t()).squeeze()
