import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import faiss
import numpy as np
from modules import *
from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    """
    k-NN + MEAN Aggregation for Dense Graphs (FB15k/Atomic)
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
        self.knn_alpha = knn_alpha  # self + knn_mean interpolation

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(dropout)

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
        
        # k-NN Components
        self.knn_index = None
        self.knn_neighbors = None  # [N_entities, knn_k]
        if knn_path is not None:
            self.load_knn_index(knn_path)
            logging.info(f'LOADED k-NN INDEX: k={knn_k}, path={knn_path}')

    def joint_cosine_similarity(self, emb1, emb2):
        """Joint cosine for ComplEx embeddings"""
        # emb1, emb2: [B, embed_dim] (real representation of complex)
        norm1 = torch.norm(emb1, dim=-1, keepdim=True)
        norm2 = torch.norm(emb2, dim=-1, keepdim=True)
        return F.cosine_similarity(emb1, emb2, dim=-1)

    def build_knn_index(self, embed_np):
        """Precompute FAISS index for fast k-NN lookup"""
        logging.info('BUILDING k-NN INDEX...')
        embeddings = torch.from_numpy(embed_np).float().cuda()
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, dim=1)
        embeddings_np = embeddings.cpu().numpy()
        
        # FAISS GPU Index (InnerProduct = cosine after norm)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, self.embed_dim)
        index.add(embeddings_np.astype('float32'))
        
        self.knn_index = index
        logging.info(f'k-NN INDEX BUILT: {embeddings_np.shape[0]} entities')

    def precompute_knn_neighbors(self):
        """Precompute all top-K neighbors for faster training"""
        if self.knn_index is None:
            raise ValueError('Must build_knn_index first')
            
        N = self.symbol_emb.weight.shape[0] - 1  # -1 for pad
        query_emb = F.normalize(self.symbol_emb.weight[:N], dim=1).cpu().numpy()
        
        distances, indices = self.knn_index.search(query_emb.astype('float32'), self.knn_k)
        self.knn_neighbors = torch.from_numpy(indices).long()
        logging.info(f'PRECOMPUTED k-NN: {self.knn_neighbors.shape}')

    def load_knn_index(self, knn_path):
        """Load precomputed k-NN neighbors"""
        self.knn_neighbors = torch.load(knn_path)
        self.knn_k = self.knn_neighbors.shape[1]
        logging.info(f'LOADED PRECOMPUTED k-NN: {self.knn_neighbors.shape}')

    def knn_neighbor_encoder(self, entity_ids):
        """Get k-NN neighbors for batch of entities"""
        if self.knn_neighbors is None:
            # Fallback to structural neighbors
            return None
            
        batch_size = entity_ids.size(0)
        knn_idx = self.knn_neighbors[entity_ids]  # [B, k]
        
        # Dummy relations (use entity-entity similarity)
        knn_rels = torch.zeros_like(knn_idx)
        knn_ents = knn_idx
        
        rel_embeds = self.dropout(self.symbol_emb(knn_rels))
        ent_embeds = self.dropout(self.symbol_emb(knn_ents))
        
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)
        out = torch.sum(out, dim=1) / self.knn_k  # YOUR ORIGINAL MEAN LOGIC
        knn_mean = out.tanh()
        
        return knn_mean

    def neighbor_encoder(self, connections, num_neighbors, entity_ids=None):
            num_neighbors = num_neighbors.unsqueeze(1).clamp(min=1)
            relations = connections[:,:,0].squeeze(-1)
            entities = connections[:,:,1].squeeze(-1)
            
            rel_embeds = self.dropout(self.symbol_emb(relations))
            ent_embeds = self.dropout(self.symbol_emb(entities))
            
            concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
            out = self.gcn_w(concat_embeds)
            
            # BACK TO MEAN: Research shows medical nodes need neighbor averaging
            out = torch.sum(out, dim=1) / num_neighbors
            structural_repr = out.tanh()
    
            # k-NN interpolation (KEEP THIS - it's your strongest feature)
            knn_mean = None
            if entity_ids is not None and self.knn_neighbors is not None:
                knn_mean = self.knn_neighbor_encoder(entity_ids)
    
            if knn_mean is not None:
                final = self.knn_alpha * structural_repr + (1 - self.knn_alpha) * knn_mean
            else:
                final = structural_repr
            return final

    def forward(self, query, support, query_meta=None, support_meta=None, entity_ids=None):
        '''
        query: (batch_size, 2)
        support: (few, 2) 
        query_meta: 6-tuple or 4-tuple
        entity_ids: [query_entity_ids, support_entity_ids] for k-NN
        '''
        if query_meta is None or support_meta is None:
            # Original fallback
            q_emb = self.symbol_emb(query).view(query.size(0), -1)
            s_emb = self.symbol_emb(support).view(support.size(0), -1)
            s_mean = s_emb.mean(dim=0, keepdim=True)
            return F.cosine_similarity(q_emb, s_mean.expand_as(q_emb))

        # Handle 6-tuple or 4-tuple meta
        if len(query_meta) == 6:
            q_l1, q_l2, q_deg_l, q_r1, q_r2, q_deg_r = query_meta
            s_l1, s_l2, s_deg_l, s_r1, s_r2, s_deg_r = support_meta
            query_left_connections, query_left_degrees = q_l1, q_deg_l
            query_right_connections, query_right_degrees = q_r1, q_deg_r
            support_left_connections, support_left_degrees = s_l1, s_deg_l
            support_right_connections, support_right_degrees = s_r1, s_deg_r
        else:
            query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
            support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        # Extract entity IDs for k-NN (head/tail)
        if entity_ids is not None:
            q_h_ids, q_t_ids = query[:,0], query[:,1]
            s_h_ids, s_t_ids = support[:,0], support[:,1]
        else:
            q_h_ids = q_t_ids = s_h_ids = s_t_ids = None

        # Neighbor encoding WITH k-NN
        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees, q_h_ids)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees, q_t_ids)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees, s_h_ids)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees, s_t_ids)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)

        # Original matching network
        support_g = self.support_encoder(support_neighbor.unsqueeze(0))
        support_g = torch.mean(support_g, dim=1)
        query_g = self.support_encoder(query_neighbor.unsqueeze(1))
        query_g = query_g.squeeze(1)
        query_f = self.query_encoder(support_g, query_g)
        
        matching_scores = torch.matmul(query_f, support_g.squeeze(0).t()).squeeze(-1)
        return matching_scores

if __name__ == '__main__':
    query = torch.ones(64,2).long() * 100
    support = torch.ones(1,2).long()  # One-shot
    matcher = EmbedMatcher(200, 40000, use_pretrain=False, knn_k=32)
    dummy_meta = [torch.zeros(64,50,2).long(), torch.ones(64)] * 3  # 6-tuple
    print(matcher(query, support, dummy_meta, dummy_meta).size())
