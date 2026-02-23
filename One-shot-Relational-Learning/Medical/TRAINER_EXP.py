import json, logging, os, random, numpy as np, torch
import torch.nn.functional as F
from torch import optim, nn
from collections import defaultdict, deque
from tqdm import tqdm
from tensorboardX import SummaryWriter
from args import read_options
from data_loader import *
from MATCHER import EmbedMatcher

class Trainer(object):
    def __init__(self, args):
        super().__init__()
        for k, v in vars(args).items(): setattr(self, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        
        # --- DATASET DETECTOR ---
        self.is_medical = 'medical' in self.dataset.lower()
        self.meta = not self.no_meta
        
        # Set file paths based on dataset type
        if self.is_medical:
            self.train_file = 'medical_object_only.train.jsonl' if self.object_only else 'medical_cleaned.train.jsonl'
        else:
            self.train_file = 'train_tasks.json'
        
        self.val_file = 'validation_tasks.json'
        self.test_file = 'test_tasks.json'

        self.ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))
        self.num_ents = len(self.ent2id)
        self.rel2candidates = json.load(open(os.path.join(self.dataset, 'rel2candidates.json')))
        self.e1rel_e2 = json.load(open(os.path.join(self.dataset, 'e1rel_e2.json')))
        
        # Valid Object Properties only (Filter out Attributes for Medical)
        self.valid_relations = set(self.rel2candidates.keys())

        if self.test or self.random_embed:
            self.load_symbol2id(); self.use_pretrain = False
        else:
            self.load_embed(); self.use_pretrain = True

        self.num_symbols = len(self.symbol2id) - 1
        self.pad_id = self.num_symbols

        # --- MATCHER SETUP ---
        knn_name = f"{self.prefix}_knn_neighbors.pt"
        knn_path = os.path.join(os.path.dirname(self.dataset), knn_name)
        
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, embed=self.symbol2vec, 
                                    knn_k=self.knn_k, knn_path=knn_path if os.path.exists(knn_path) else None)
        if torch.cuda.device_count() > 1: self.matcher = nn.DataParallel(self.matcher)
        self.matcher.to(self.device)

        # Split LR for Gating Stability
        m_obj = self.matcher.module if hasattr(self.matcher, 'module') else self.matcher
        gate_params = list(m_obj.gate_layer.parameters())
        base_params = [p for n, p in m_obj.named_parameters() if 'gate_layer' not in n]
        
        self.optim = optim.Adam([{'params': base_params, 'lr': self.lr}, 
                                 {'params': gate_params, 'lr': self.lr * 0.1}], weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.5)

        self.build_connection_matrix(max_=self.max_neighbor)
        self.batch_nums = 0
        self.writer = None if self.test else SummaryWriter('logs/' + self.prefix)
        print(f"Unified Trainer initialized. Mode: {'Medical' if self.is_medical else 'NELL'}")

    def load_embed(self):
        rel2id = json.load(open(os.path.join(self.dataset, 'relation2ids')))
        ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))
        ent_embed = np.loadtxt(os.path.join(self.dataset, f'entity2vec.{self.embed_model}'))
        rel_embed = np.loadtxt(os.path.join(self.dataset, f'relation2vec.{self.embed_model}'))
        ent_embed = (ent_embed - ent_embed.mean()) / (ent_embed.std() + 1e-3)
        rel_embed = (rel_embed - rel_embed.mean()) / (rel_embed.std() + 1e-3)

        if getattr(self, 'use_semantic', False):
            sb_file = os.path.join(os.path.dirname(self.dataset), f"{self.prefix}_anchors.npy")
            if os.path.exists(sb_file):
                sb = np.load(sb_file)
                sb = (sb - sb.mean(0)) / (sb.std(0) + 1e-3)
                if sb.shape[1] != ent_embed.shape[1]:
                    proj = np.random.RandomState(42).normal(0, 1.0/np.sqrt(sb.shape[1]), (sb.shape[1], ent_embed.shape[1]))
                    sb = sb @ proj
                ent_embed = np.concatenate([ent_embed, sb], axis=1)
                rel_embed = np.concatenate([rel_embed, np.zeros((rel_embed.shape[0], sb.shape[1]))], axis=1)

        embeddings = []
        self.symbol2id = {k: i for i, k in enumerate(list(rel2id.keys()) + list(ent2id.keys()))}
        for k in rel2id.keys(): embeddings.append(list(rel_embed[rel2id[k]]))
        for k in ent2id.keys(): embeddings.append(list(ent_embed[ent2id[k]]))
        embeddings.append(np.zeros(ent_embed.shape[1])) # PAD
        self.symbol2vec = np.array(embeddings)

    def load_tasks(self, path):
        if path.endswith('.json'): return json.load(open(path))
        tasks = defaultdict(list)
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                h, r, t = d.get('h') or d.get('head'), d.get('r') or d.get('relation'), d.get('t') or d.get('tail')
                if h and r and t: tasks[r].append([h, r, t])
        return tasks

    def build_connection_matrix(self, max_=100):
        self.connections = np.ones((self.num_ents, max_, 2), dtype=int) * (len(self.symbol2id)-1)
        self.e1_rele2 = defaultdict(list)
        with open(os.path.join(self.dataset, 'path_graph')) as f:
            for line in f:
                e1, rel, e2 = line.rstrip().split()
                if rel in self.symbol2id and e1 in self.symbol2id and e2 in self.symbol2id:
                    self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                    inv = rel if rel.endswith('_inv') else rel + '_inv'
                    if inv in self.symbol2id: self.e1_rele2[e2].append((self.symbol2id[inv], self.symbol2id[e1]))
        
        for ent, idx in self.ent2id.items():
            neighbors = self.e1_rele2.get(ent, [])[:max_]
            for j, (rid, eid) in enumerate(neighbors):
                self.connections[idx, j, 0], self.connections[idx, j, 1] = rid, eid
        self.connections = torch.LongTensor(self.connections).to(self.device)
        self.e1_degrees_tensor = torch.FloatTensor([float(len(self.e1_rele2.get(e, []))) for e in self.ent2id.keys()]).to(self.device)

    def get_meta(self, left, right):
        l_idx, r_idx = torch.LongTensor(left).to(self.device), torch.LongTensor(right).to(self.device)
        return (self.connections[l_idx], self.e1_degrees_tensor[l_idx], self.connections[r_idx], self.e1_degrees_tensor[r_idx])

    def train(self):
        logging.info("STARTING TRAINING")
        best_hits10 = 0.0
        rel_freq = defaultdict(int)
        train_tasks = self.load_tasks(os.path.join(self.dataset, self.train_file))
        for r, triples in train_tasks.items(): 
            if not self.is_medical or r in self.valid_relations: rel_freq[r] = len(triples)
        
        max_f = max(rel_freq.values()) if rel_freq else 1
        weight_tensor = torch.ones(len(self.symbol2id), device=self.device)
        for r, f in rel_freq.items():
            rid = self.symbol2id.get(self.escape_token(r), self.pad_id)
            weight_tensor[rid] = min(max((max_f / (f + 1e-6))**0.5, 1.0), 10.0)

        loader = train_generate_medical if self.is_medical else train_generate
        for data in loader(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2):
            if self.is_medical: s_p, q_p, f_p, s_l, s_r, q_l, q_r, f_l, f_r, rel_name = data
            else: 
                s_p, q_p, f_p, s_l, s_r, q_l, q_r, f_l, f_r = data
                rel_name = "nell_task"

            if self.is_medical and rel_name not in self.valid_relations: continue

            q_scores = self.matcher(torch.LongTensor(q_p).to(self.device), torch.LongTensor(s_p).to(self.device), self.get_meta(q_l, q_r), self.get_meta(s_l, s_r))
            f_scores = self.matcher(torch.LongTensor(f_p).to(self.device), torch.LongTensor(s_p).to(self.device), self.get_meta(f_l, f_r), self.get_meta(s_l, s_r))

            # Adversarial Loss logic (Softmax)
            f_scores = f_scores.view(q_scores.size(0), -1)
            adv_w = F.softmax(f_scores, dim=-1).detach()
            adv_f = (adv_w * f_scores).sum(dim=-1)
            
            rid = self.symbol2id.get(self.escape_token(rel_name), self.pad_id)
            loss = (F.relu(self.margin - (q_scores - adv_f)) * weight_tensor[rid]).mean()
            
            self.optim.zero_grad(); loss.backward(); self.optim.step()
            if self.batch_nums % self.eval_every == 0 and self.batch_nums > 0:
                h10, _, mrr, _ = self.eval(mode='dev', meta=self.meta)
                if h10 > best_hits10: self.save(self.save_path + "_bestHits10"); best_hits10 = h10
            self.batch_nums += 1
            if self.batch_nums >= self.max_batches: break

    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        hits10, mrr = [], []
        test_tasks = self.load_tasks(os.path.join(self.dataset, self.val_file if mode == 'dev' else self.test_file))
        for r_name, triples in test_tasks.items():
            if (self.is_medical and r_name not in self.valid_relations) or len(triples) < 2: continue
            candidates = self.rel2candidates[r_name]
            support_p = [[self.symbol2id[t[0]], self.symbol2id[t[2]]] for t in triples[:self.few]]
            support_meta = self.get_meta([self.ent2id[t[0]] for t in triples[:self.few]], [self.ent2id[t[2]] for t in triples[:self.few]])
            
            for tri in triples[self.few:]:
                valid_c = [c for c in candidates if c in self.ent2id]
                if tri[2] not in valid_c: valid_c.append(tri[2])
                q_batch = torch.LongTensor([[self.symbol2id[tri[0]], self.symbol2id[c]] for c in valid_c]).to(self.device)
                q_meta = self.get_meta([self.ent2id[tri[0]]] * len(valid_c), [self.ent2id[c] for c in valid_c])
                with torch.no_grad(): scores = self.matcher(q_batch, torch.LongTensor(support_p).to(self.device), q_meta, support_meta)
                rank = (scores > scores[valid_c.index(tri[2])]).sum().item() + 1
                hits10.append(1.0 if rank <= 10 else 0.0); mrr.append(1.0 / rank)
        
        self.matcher.train(); res = np.mean(hits10)
        logging.critical(f"Mode {mode} Hits@10: {res:.3f} MRR: {np.mean(mrr):.3f}")
        return res, 0, np.mean(mrr), {}

    def escape_token(self, token):
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(token))

if __name__ == "__main__":
    args = read_options(); trainer = Trainer(args)
    if args.test: trainer.eval(mode='test', meta=trainer.meta)
    else: trainer.train()
