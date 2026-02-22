import json
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import defaultdict, deque
from tqdm import tqdm
from tensorboardX import SummaryWriter

from args import read_options
from data_loader import *
from MATCHER import EmbedMatcher

class Trainer(object):
    def __init__(self, args):
        super().__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
    
        # --- DEVICE ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logging.warning("No CUDA found. Running on CPU.")
        torch.backends.cudnn.benchmark = True
    
        # --- METADATA ---
        self.meta = not self.no_meta
        self.use_pretrain = not self.random_embed
    
        # --- DATASET FILES ---
        self.train_file = 'train_tasks.json'
        self.val_file = 'validation_tasks.json'
        self.test_file = 'test_tasks.json'
    
        self.ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))
        self.num_ents = len(self.ent2id)
        self.rel2candidates = json.load(open(os.path.join(self.dataset, 'rel2candidates.json')))
        self.e1rel_e2 = json.load(open(os.path.join(self.dataset, 'e1rel_e2.json')))
    
        # --- LOAD SYMBOLS / EMBEDDINGS ---
        logging.info("LOADING SYMBOLS AND EMBEDDINGS")
        if self.test or self.random_embed:
            self.load_symbol2id()
            self.use_pretrain = False
        else:
            self.load_embed()
    
        # --- SEMANTIC .NPY PATH (NELL-One) ---
        if hasattr(self, 'use_semantic') and self.use_semantic:
            self.semantic_path = os.path.join(
                os.path.dirname(self.dataset), f"{self.prefix}_anchors.npy"
            )
            if not os.path.exists(self.semantic_path):
                logging.warning(f"Semantic .npy not found at {self.semantic_path}")
    
        self.num_symbols = len(self.symbol2id) - 1
        self.pad_id = self.num_symbols
    
        # --- MATCHER ---
        self.matcher = EmbedMatcher(
            embed_dim=self.embed_dim,
            num_symbols=self.num_symbols,
            use_pretrain=self.use_pretrain,
            embed=self.symbol2vec,
            dropout=self.dropout,
            batch_size=self.batch_size,
            process_steps=self.process_steps,
            finetune=self.fine_tune,
            aggregate=self.aggregate,
            knn_k=self.knn_k,
        )
        if torch.cuda.device_count() > 1:
            self.matcher = nn.DataParallel(self.matcher)
        self.matcher.to(self.device)
    
        # --- OPTIMIZER with Gate LR ---
        if torch.cuda.device_count() > 1:
            gate_params = list(self.matcher.module.gate_layer.parameters())
            base_params = [p for n, p in self.matcher.module.named_parameters() if 'gate_layer' not in n]
        else:
            gate_params = list(self.matcher.gate_layer.parameters())
            base_params = [p for n, p in self.matcher.named_parameters() if 'gate_layer' not in n]
    
        self.optim = optim.Adam([
            {'params': base_params, 'lr': self.lr},
            {'params': gate_params, 'lr': self.lr * 0.1}
        ], weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.5)
    
        # --- CLAMP NEIGHBORS TO DATASET SIZE ---
        self.max_neighbor = min(self.max_neighbor, self.num_ents)
        self.knn_k = min(self.knn_k, self.max_neighbor)
    
        # --- CONNECTION MATRIX ---
        self.connections = None
        self.e1_degrees_tensor = None
        self.build_connection_matrix(max_=self.max_neighbor)
    
        # --- LOGGER & TENSORBOARD ---
        self.batch_nums = 0
        self.writer = None if self.test else SummaryWriter('logs/' + self.prefix)
    
        print(f"Trainer initialized on {self.device}, {torch.cuda.device_count()} GPUs.")
    # ---------------- SYMBOLS / EMBEDDINGS ----------------
    def load_symbol2id(self):
        symbol_id = {}
        rel2id = json.load(open(os.path.join(self.dataset, 'relation2ids')))
        ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))
        i = 0
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):
        rel2id = json.load(open(os.path.join(self.dataset, 'relation2ids')))
        ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))
    
        ent_file = os.path.join(self.dataset, f'entity2vec.{self.embed_model}')
        rel_file = os.path.join(self.dataset, f'relation2vec.{self.embed_model}')
    
        ent_embed = np.loadtxt(ent_file)
        rel_embed = np.loadtxt(rel_file)
    
        # --- STANDARDIZE ---
        ent_embed = (ent_embed - ent_embed.mean()) / (ent_embed.std() + 1e-3)
        rel_embed = (rel_embed - rel_embed.mean()) / (rel_embed.std() + 1e-3)
    
        # --- SEMANTIC CHANNEL ---
        if getattr(self, 'use_semantic', False):
            sb_file = getattr(self, 'semantic_path', os.path.join(os.path.dirname(self.dataset), f'{self.prefix}_anchors.npy'))
            if os.path.exists(sb_file):
                sb = np.load(sb_file)
                sb = (sb - sb.mean(axis=0, keepdims=True)) / (sb.std(axis=0, keepdims=True) + 1e-3)
                if sb.shape[1] != ent_embed.shape[1]:
                    rng = np.random.RandomState(42)
                    proj = rng.normal(0, 1.0 / np.sqrt(sb.shape[1]), size=(sb.shape[1], ent_embed.shape[1]))
                    sb = sb @ proj
                ent_embed = np.concatenate([ent_embed, sb], axis=1)
                rel_embed = np.concatenate([rel_embed, np.zeros((rel_embed.shape[0], sb.shape[1]))], axis=1)
                logging.info(f'Unified embedding dim: {ent_embed.shape[1]}')
            else:
                logging.warning(f'Semantic .npy not found at {sb_file}')
    
        # --- FINAL SYMBOL EMBEDDINGS ---
        symbol_id = {}
        embeddings = []
        i = 0
    
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(rel_embed[rel2id[key]]))
    
        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(ent_embed[ent2id[key]]))
    
        symbol_id['PAD'] = i
        embeddings.append(np.zeros(ent_embed.shape[1]))
    
        self.symbol2id = symbol_id
        self.symbol2vec = np.array(embeddings)

    # ---------------- CONNECTION MATRIX ----------------
    def build_connection_matrix(self, max_=100):
        self.connections = np.ones((self.num_ents, max_, 2), dtype=int) * self.pad_id
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)

        with open(os.path.join(self.dataset, 'path_graph')) as f:
            for line in tqdm(f.readlines(), desc="Building Connections"):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                inv_rel = rel if rel.endswith('_inv') else rel + '_inv'
                self.e1_rele2[e2].append((self.symbol2id[inv_rel], self.symbol2id[e1]))

        for ent, idx in self.ent2id.items():
            neighbors = self.e1_rele2.get(ent, [])[:max_]
            self.e1_degrees[idx] = len(neighbors)
            for j, (rid, eid) in enumerate(neighbors):
                self.connections[idx, j, 0] = rid
                self.connections[idx, j, 1] = eid

        self.connections = torch.LongTensor(self.connections).to(self.device)
        degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(degrees_list).to(self.device)

    # ---------------- META ----------------
    def get_meta(self, left, right):
        left_idx = torch.LongTensor(left).to(self.device)
        right_idx = torch.LongTensor(right).to(self.device)
        left_connections = self.connections[left_idx]
        left_degrees = self.e1_degrees_tensor[left_idx]
        right_connections = self.connections[right_idx]
        right_degrees = self.e1_degrees_tensor[right_idx]
        return (left_connections, left_degrees, right_connections, right_degrees)

    # ---------------- TRAIN ----------------
    def train(self):
        logging.info("STARTING TRAINING")
        best_hits10 = 0.0
        losses = deque(maxlen=self.log_every)

        # TASK WEIGHTS
        relation_freq = defaultdict(int)
        train_path = os.path.join(self.dataset, self.train_file)
        with open(train_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                rel = data.get('relation') or data.get('r')
                if rel: relation_freq[rel] += 1
        max_freq = max(relation_freq.values()) if relation_freq else 1
        weight_tensor = torch.ones(len(self.symbol2id), device=self.device)
        for rel, freq in relation_freq.items():
            rid = self.symbol2id.get(rel, self.pad_id)
            w = min(max((max_freq / (freq + 1e-6))**0.5, 1.0), 10.0)
            weight_tensor[rid] = w

        # DATA LOADER
        from data_loader import train_generate_medical  # reused loader for both datasets
        for data in train_generate_medical(self.dataset, self.batch_size, self.train_few,
                                           self.symbol2id, self.ent2id, self.e1rel_e2,
                                           train_file=self.train_file):
            support_p, query_p, false_p, s_l, s_r, q_l, q_r, f_l, f_r, rel_name = data

            support_meta = tuple(t.to(self.device) for t in self.get_meta(s_l, s_r))
            query_meta = tuple(t.to(self.device) for t in self.get_meta(q_l, q_r))
            false_meta = tuple(t.to(self.device) for t in self.get_meta(f_l, f_r))

            support = torch.clamp(torch.LongTensor(support_p).to(self.device), 0, self.pad_id)
            query = torch.clamp(torch.LongTensor(query_p).to(self.device), 0, self.pad_id)
            false = torch.clamp(torch.LongTensor(false_p).to(self.device), 0, self.pad_id)

            esc_rel_name = rel_name
            rel_id = self.symbol2id.get(esc_rel_name, self.pad_id)
            task_weight = weight_tensor[rel_id] if rel_id < len(weight_tensor) else 1.0

            if self.no_meta:
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
                query_scores = self.matcher(query, support, query_meta, support_meta)
                false_scores = self.matcher(false, support, false_meta, support_meta)

            # ADVERSARIAL NEGATIVE SAMPLING
            adv_temperature = 1.0
            neg_num = false_scores.size(0) // query_scores.size(0)
            f_scores = false_scores.view(query_scores.size(0), neg_num)
            adv_weights = F.softmax(f_scores * adv_temperature, dim=-1)
            adv_false_scores = (adv_weights * f_scores).sum(dim=-1)

            loss = (F.relu(self.margin - (query_scores - adv_false_scores)) * task_weight).mean()
            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.batch_nums % self.log_every == 0:
                logging.critical(f"Batch {self.batch_nums}: Loss={np.mean(losses):.4f} (Task {rel_name}, Weight {task_weight:.1f})")

            if self.batch_nums % self.eval_every == 0 and self.batch_nums > 0:
                hits10, hits5, mrr, _ = self.eval(mode='dev', meta=self.meta)
                self.save()
                if hits10 > best_hits10:
                    self.save(self.save_path + "_bestHits10")
                    best_hits10 = hits10

            self.batch_nums += 1
            self.scheduler.step()
            if self.batch_nums >= self.max_batches:
                hits10, hits5, mrr, _ = self.eval(mode='dev', meta=self.meta)
                logging.critical(f"FINAL RESULTS - HITS@10: {hits10:.3f}, MRR: {mrr:.3f}")
                self.save()
                break

    # ---------------- SAVE / LOAD ----------------
    def save(self, path="models/initial"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(self.matcher, nn.DataParallel):
            torch.save(self.matcher.module.state_dict(), path)
        else:
            torch.save(self.matcher.state_dict(), path)

    def load(self):
        state = torch.load(self.save_path, map_location=self.device)
        if isinstance(self.matcher, nn.DataParallel):
            self.matcher.module.load_state_dict(state)
        else:
            self.matcher.load_state_dict(state)

    # ---------------- EVALUATION ----------------
    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        hits10, hits5, hits1, mrr = [], [], [], []
    
        task_path = os.path.join(self.dataset, self.val_file if mode == 'dev' else self.test_file)
        test_tasks = self.load_tasks(task_path)
    
        for rel_name, triples in test_tasks.items():
            if len(triples) < 2:
                continue
    
            # get candidate tails for this relation
            raw_candidates = self.rel2candidates.get(rel_name, [])
            if not raw_candidates:
                logging.warning(f"No candidates for relation {rel_name}; skipped.")
                continue
    
            # build support set
            support_triples = triples[:self.few]
            support_pairs = [[self.symbol2id[t[0]], self.symbol2id[t[2]]] for t in support_triples]
    
            support_left = [self.ent2id[t[0]] for t in support_triples]
            support_right = [self.ent2id[t[2]] for t in support_triples]
            support_meta = self.get_meta(support_left, support_right)
            support_meta = tuple(t.to(self.device) for t in support_meta)
            support = torch.LongTensor(support_pairs).to(self.device)
    
            for triple in triples[self.few:]:
                # collect valid candidate list
                valid_cands = []
                for ent in raw_candidates:
                    # filter true match presence in graph
                    if ent == triple[2]:
                        valid_cands.append(ent)
                    else:
                        # if present in graph but not forbidden
                        if (triple[0] in self.e1rel_e2 and
                            rel_name in self.e1rel_e2[triple[0]] and
                            ent in self.e1rel_e2[triple[0]][rel_name]):
                            continue
                        valid_cands.append(ent)
    
                if not valid_cands:
                    logging.warning(f"No valid candidates for query {triple} under relation {rel_name}")
                    continue  # skip this query, no ranking possible
    
                # build batch
                batch_syms = [[self.symbol2id[triple[0]], self.symbol2id[c]] for c in valid_cands]
                query = torch.LongTensor(batch_syms).to(self.device)
    
                query_meta = None
                if meta:
                    query_left = [self.ent2id[triple[0]]]*len(valid_cands)
                    query_right = [self.ent2id[c]   for c in valid_cands]
                    query_meta = tuple(t.to(self.device) for t in self.get_meta(query_left, query_right))
    
                # score candidates
                scores = self.matcher(query, support, query_meta, support_meta)
    
                if scores.numel() == 0:
                    logging.warning(f"Empty scores for relation {rel_name} with query {triple}")
                    continue
    
                true_idx = valid_cands.index(triple[2])
                true_score = scores[true_idx].item()
    
                # compute rank
                rank = (scores > true_score).sum().item() + 1
    
                hits10.append(1.0 if rank <= 10 else 0.0)
                hits5.append(1.0 if rank <= 5 else 0.0)
                hits1.append(1.0 if rank <= 1 else 0.0)
                mrr.append(1.0 / rank)
    
        logging.critical(f'OVERALL HITS@10: {np.mean(hits10):.3f}, MRR: {np.mean(mrr):.3f}')
        self.matcher.train()
        return np.mean(hits10), np.mean(hits5), np.mean(mrr), None

    def load_tasks(self, file_path):
        if file_path.endswith('.json'):
            return json.load(open(file_path))
        tasks = defaultdict(list)
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                h = data.get('head') or data.get('h')
                r = data.get('relation') or data.get('r')
                t = data.get('tail') or data.get('t')
                if h and r and t:
                    tasks[r].append([h, r, t])
        return tasks

    def escape_token(self, token):
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(token))

if __name__ == "__main__":
    args = read_options()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(args)
    if args.test:
        trainer.eval(meta=trainer.meta)
    else:
        trainer.train()
