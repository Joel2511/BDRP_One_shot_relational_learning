import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

# === SAFER CHANGE START ===
class SAFERSupportAdaptation(nn.Module):
    def __init__(self, embed_dim, num_iters=1):
        super(SAFERSupportAdaptation, self).__init__()
        self.embed_dim = embed_dim
        self.num_iters = num_iters
        # neural net for edge weighting and adaptation
        self.edge_nn = nn.Linear(embed_dim, embed_dim)

    def forward(self, support_graphs):
        # support_graphs: [num_support, num_edges, embed_dim]
        adapted_graphs = []
        for sg in support_graphs:
            edge_embeds = self.edge_nn(sg) # shape: [num_edges, embed_dim]
            weights = torch.sigmoid(edge_embeds.mean(dim=-1, keepdim=True)) # [num_edges, 1]
            adapted = sg * weights  # edge-wise multiplication
            adapted_graphs.append(adapted)
        return adapted_graphs
# === SAFER CHANGE END ===

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

        self.dropout = nn.Dropout(0.5)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            # emb_np = np.loadtxt(embed_path)
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

        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''

        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1) # tanh
        support_neighbor = torch.cat((support_left, support_right), dim=-1) # tanh

        support = support_neighbor
        query = query_neighbor

        support_g = self.support_encoder(support) # 1 * 100
        query_g = self.support_encoder(query)

        support_g = torch.mean(support_g, dim=0, keepdim=True)
        # support_g = support
        # query_g = query

        query_f = self.query_encoder(support_g, query_g) # 128 * 100

        # cosine similarity
        matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

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

# ... rest of RescalMatcher and other classes (unchanged)
