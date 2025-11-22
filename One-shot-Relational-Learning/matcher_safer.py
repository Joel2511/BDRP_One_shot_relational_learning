import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from module_safer import SAFERSupportAdaptation
from modules import Path
from torch.autograd import Variable

# === SAFER MATCHER ===
class SAFERMatcher(nn.Module):
    """
    SAFER Matching metric based on subgraph adaptation
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.2, k_num=100):
        super(SAFERMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.k_num = k_num

        self.dropout = nn.Dropout(dropout)

        if use_pretrain and embed is not None:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            self.symbol_emb.weight.requires_grad = False

        # SAFER adaptation networks
        self.support_adaptor = SAFERSupportAdaptation(embed_dim)
        self.query_encoder = nn.Linear(embed_dim, embed_dim)  # you can extend to more complex

        # Path convolution (as in your Path module)
        self.path_conv = Path(input_dim=embed_dim, num_symbols=num_symbols, use_pretrain=False, k_sizes=[3], k_num=self.k_num)

    def encode_paths(self, path_indices):
        """
        path_indices: list of [num_paths, max_len]
        Returns:
            tensor: [num_paths, embed_dim]
        """
        if len(path_indices) == 0:
            # handle empty
            return torch.zeros(1, self.embed_dim).cuda()
        paths_tensor = torch.LongTensor(path_indices).cuda()
        path_embeds = self.path_conv(paths_tensor) # [num_paths, embed_dim]
        return path_embeds

    def forward(self, support_subgraphs, query_subgraphs, false_subgraphs=None):
        '''
        Each input is: list of list of path indices (for SAFER: subgraphs are list of paths as indices)
        '''
        # Encode and adapt support subgraphs
        support_embeds_all = []
        for support_paths in support_subgraphs:
            if len(support_paths) == 0:
                continue
            support_embeds_all.append(self.encode_paths(support_paths))
        # [num_support, num_paths, embed_dim] → support adaptation
        support_adapted = self.support_adaptor(support_embeds_all)
        # mean-pool over all adapted support paths, then average across supports
        support_repr = torch.stack([sa.mean(dim=0) for sa in support_adapted]).mean(dim=0) # [embed_dim,]

        # Encode query subgraphs
        query_embeds_all = []
        for query_paths in query_subgraphs:
            if len(query_paths) == 0:
                continue
            query_embeds_all.append(self.encode_paths(query_paths))
        # mean-pool over all query paths (for each test query)
        query_repr = torch.stack([qe.mean(dim=0) for qe in query_embeds_all]) # [batch_size, embed_dim]
        query_repr = self.query_encoder(query_repr) # (optional SAE block)

        # Cosine similarity for SAFER
        similarity = F.cosine_similarity(query_repr, support_repr.unsqueeze(0), dim=1) # [batch_size]
        return similarity

    # Optionally implement negative scoring (if using negatives for loss)
    def score_negatives(self, support_subgraphs, false_subgraphs):
        # Analogous to forward(), encode false_subgraphs vs support_subgraphs
        false_embeds_all = []
        for false_paths in false_subgraphs:
            if len(false_paths) == 0:
                continue
            false_embeds_all.append(self.encode_paths(false_paths))
        false_repr = torch.stack([fe.mean(dim=0) for fe in false_embeds_all])
        return false_repr

if __name__ == '__main__':
    # Minimal test code: (paths must be pre-extracted/encoded)
    # For real usage, supply actual subgraph path indices
    matcher = SAFERMatcher(100, 10000, use_pretrain=False)
    support_subgraphs = [ [[1,2,3,4], [5,6,7,8]], [[9,2,7,5]], [[2,3,4,0]] ] # example
    query_subgraphs = [ [[1,2,3,4], [5,6,7]] ]
    out = matcher(support_subgraphs, query_subgraphs)
    print(out.size())
