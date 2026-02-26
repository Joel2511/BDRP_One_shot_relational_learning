import json
import logging
import os
import re
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
from data_loader import train_generate_medical
from MATCHER_EXP import EmbedMatcher


def is_medical(prefix):
    return 'medical' in prefix.lower()

def is_atomic(prefix):
    return 'atomic' in prefix.lower()


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

        self.meta      = not self.no_meta
        self._is_medical = is_medical(self.prefix)
        self._is_atomic  = is_atomic(self.prefix)

        # --- DATASET FILES ---
        if self._is_medical:
            if self.object_only:
                self.train_file = 'medical_object_only.train.jsonl'
                self.val_file   = 'medical_object_only.validation.jsonl'
                self.test_file  = 'medical_object_only.test.jsonl'
            else:
                self.train_file = 'medical_cleaned.train.jsonl'
                self.val_file   = 'validation_tasks.json'
                self.test_file  = 'test_tasks.json'
        elif self._is_atomic:
            self.train_file = 'train_tasks.json'
            self.val_file   = 'dev_tasks.json'
            self.test_file  = 'test_tasks.json'
        else:
            self.train_file = 'train_tasks.json'
            self.val_file   = 'validation_tasks.json'
            self.test_file  = 'test_tasks.json'

        # --- LOAD ENTITY/RELATION MAPS ---
        self.ent2id         = json.load(open(os.path.join(self.dataset, 'ent2ids')))
        self.num_ents       = len(self.ent2id)
        self.rel2candidates = json.load(open(os.path.join(self.dataset, 'rel2candidates.json')))
        self.e1rel_e2       = json.load(open(os.path.join(self.dataset, 'e1rel_e2.json')))

        # --- LOAD SYMBOLS / EMBEDDINGS ---
        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        use_pretrain = not self.random_embed
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
        else:
            self.load_embed()
        self.use_pretrain = use_pretrain

        # Unified pad convention: PAD is always the last entry in symbol2id
        # num_symbols = total entries - 1 (excludes PAD)
        # pad_id      = num_symbols      (index of PAD row in embedding)
        self.num_symbols = len(self.symbol2id) - 1
        self.pad_id      = self.num_symbols

        # --- MATCHER ---
        self.matcher = EmbedMatcher(
            embed_dim=self.embed_dim,
            num_symbols=self.num_symbols,
            use_pretrain=True,
            embed=self.symbol2vec,
            dropout=self.dropout,
            batch_size=self.batch_size,
            finetune=self.fine_tune,
            semantic_matrix=self.semantic_matrix
        )
        if torch.cuda.device_count() > 1:
            self.matcher = nn.DataParallel(self.matcher)
        self.matcher.to(self.device)

        # --- OPTIMIZER: gate learns 10x slower ---
        m = self.matcher.module if isinstance(self.matcher, nn.DataParallel) else self.matcher
        gate_params = []
        base_params = [p for n, p in m.named_parameters() if 'gate_layer' not in n]
        self.optim = optim.Adam([
            {'params': base_params, 'lr': self.lr},
            {'params': gate_params, 'lr': self.lr * 0.1},
        ], weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.5)

        # --- CONNECTION MATRIX ---
        self.max_neighbor = min(self.max_neighbor, self.num_ents)
        self.knn_k        = min(self.knn_k, self.max_neighbor)
        self.connections  = None
        self.e1_degrees_tensor = None
        self.build_connection_matrix(max_=self.max_neighbor)

        # --- MISC ---
        self.batch_nums = 0
        self.writer = None if self.test else SummaryWriter('logs/' + self.prefix)
        print(f"Trainer initialized on {self.device}, {torch.cuda.device_count()} GPUs.")

    # ------------------------------------------------------------------
    # Token escaping — only needed for medical ontology URIs
    # ------------------------------------------------------------------
    def escape_token(self, token):
        if self._is_medical:
            return re.sub(r'[^a-zA-Z0-9_]', '_', str(token))
        return str(token)

    def _sym(self, token):
        return self.symbol2id.get(self.escape_token(token), self.pad_id)

    def _ent(self, token):
        return self.ent2id.get(self.escape_token(token), 0)

    # ------------------------------------------------------------------
    # Symbol / Embedding loading
    # ------------------------------------------------------------------
    def load_symbol2id(self):
        symbol_id = {}
        rel2id = json.load(open(os.path.join(self.dataset, 'relation2ids')))
        ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))
        i = 0
        for key in rel2id:
            if key not in ['', 'OOV']:
                symbol_id[self.escape_token(key)] = i
                i += 1
        for key in ent2id:
            if key not in ['', 'OOV']:
                symbol_id[self.escape_token(key)] = i
                i += 1
        symbol_id['PAD'] = i
        self.symbol2id  = symbol_id
        self.symbol2vec = None

    def load_embed(self):
        symbol_id = {}
        rel2id = json.load(open(os.path.join(self.dataset, 'relation2ids')))
        ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))
    
        for key in rel2id.keys():
            symbol_id[key] = rel2id[key]
        for key in ent2id.keys():
            symbol_id[key] = ent2id[key] + len(rel2id)
    
        self.symbol2id = symbol_id
        self.num_symbols = len(symbol_id)
    
        logging.info('LOADING PRE-TRAINED EMBEDDING')
        ent_file = os.path.join(self.dataset, 'entity2vec.' + self.embed_model)
        rel_file = os.path.join(self.dataset, 'relation2vec.' + self.embed_model)
    
        ent_embed = np.loadtxt(ent_file)
        rel_embed = np.loadtxt(rel_file)
    
        # Standardize structural embeddings
        ent_embed = (ent_embed - np.mean(ent_embed)) / (np.std(ent_embed) + 1e-3)
        rel_embed = (rel_embed - np.mean(rel_embed)) / (np.std(rel_embed) + 1e-3)
    
        self.embed_dim = ent_embed.shape[1]
    
        # Store semantic separately (NO projection, NO concatenation)
        self.semantic_matrix = None
        if getattr(self, 'use_semantic', False):
            sb_file = os.path.join(
                os.path.dirname(self.dataset),
                f'{self.prefix}_anchors.npy'
            )
    
            if os.path.exists(sb_file):
                logging.info(f'Loading semantic channel from {sb_file}')
                sb = np.load(sb_file)
                sb = (sb - sb.mean(axis=0, keepdims=True)) / (sb.std(axis=0, keepdims=True) + 1e-3)
                self.semantic_matrix = sb  # keep raw 768d
            else:
                logging.warning(f'Semantic .npy not found at {sb_file}')
    
        assert ent_embed.shape[0] == len(ent2id)
        assert rel_embed.shape[0] == len(rel2id)
    
        # Combine structural only
        rel_embed = np.concatenate([rel_embed, ent_embed], axis=0)
        pad = np.zeros((1, self.embed_dim))
        self.symbol2vec = np.concatenate([pad, rel_embed], axis=0)

    # ------------------------------------------------------------------
    # Connection matrix
    # ------------------------------------------------------------------
    def build_connection_matrix(self, max_=100):
        self.connections  = np.ones((self.num_ents, max_, 2), dtype=int) * self.pad_id
        self.e1_rele2     = defaultdict(list)
        self.e1_degrees   = defaultdict(int)

        with open(os.path.join(self.dataset, 'path_graph')) as f:
            for line in tqdm(f.readlines(), desc="Building Connections"):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self._sym(rel), self._sym(e2)))
                inv_rel = rel if rel.endswith('_inv') else rel + '_inv'
                self.e1_rele2[e2].append((self._sym(inv_rel), self._sym(e1)))

        for ent, idx in self.ent2id.items():
            neighbors = self.e1_rele2.get(ent, [])[:max_]
            self.e1_degrees[idx] = len(neighbors)
            for j, (rid, eid) in enumerate(neighbors):
                self.connections[idx, j, 0] = rid
                self.connections[idx, j, 1] = eid

        self.connections = torch.LongTensor(self.connections).to(self.device)
        degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(degrees_list).to(self.device)

    # ------------------------------------------------------------------
    # Meta helpers
    # ------------------------------------------------------------------
    def get_meta(self, left, right):
        l = torch.LongTensor(left).to(self.device)
        r = torch.LongTensor(right).to(self.device)
        return (self.connections[l], self.e1_degrees_tensor[l],
                self.connections[r], self.e1_degrees_tensor[r])

    # ------------------------------------------------------------------
    # Task weights
    # ------------------------------------------------------------------
    def compute_task_weights(self):
        relation_freq = defaultdict(int)
        train_path = os.path.join(self.dataset, self.train_file)
        try:
            if self.train_file.endswith('.json'):
                data = json.load(open(train_path))
                for rel, triples in data.items():
                    relation_freq[rel] = len(triples)
            else:
                with open(train_path, 'r') as f:
                    for line in f:
                        try:
                            d   = json.loads(line)
                            rel = d.get('relation') or d.get('r')
                            if rel:
                                relation_freq[rel] += 1
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logging.warning(f'compute_task_weights: could not parse {train_path}: {e}')

        if not relation_freq:
            return torch.ones(len(self.symbol2id), device=self.device)

        max_freq      = max(relation_freq.values())
        weight_tensor = torch.ones(len(self.symbol2id), device=self.device)
        for rel, freq in relation_freq.items():
            rid = self._sym(rel)
            if rid < len(weight_tensor):
                w = float(min(max((max_freq / (freq + 1e-6)) ** 0.5, 1.0), 10.0))
                weight_tensor[rid] = w
        return weight_tensor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        logging.info('START TRAINING...')
        best_hits10   = 0.0
        losses        = deque(maxlen=self.log_every)
        weight_tensor = self.compute_task_weights()

        for data in train_generate_medical(
            self.dataset, self.batch_size, self.train_few,
            self.symbol2id, self.ent2id, self.e1rel_e2,
            train_file=self.train_file
        ):
            support_p, query_p, false_p, s_l, s_r, q_l, q_r, f_l, f_r, rel_name = data

            support_meta = tuple(t.to(self.device) for t in self.get_meta(s_l, s_r))
            query_meta   = tuple(t.to(self.device) for t in self.get_meta(q_l, q_r))
            false_meta   = tuple(t.to(self.device) for t in self.get_meta(f_l, f_r))

            support = torch.clamp(torch.LongTensor(support_p).to(self.device), 0, self.pad_id)
            query   = torch.clamp(torch.LongTensor(query_p).to(self.device),   0, self.pad_id)
            false   = torch.clamp(torch.LongTensor(false_p).to(self.device),   0, self.pad_id)

            rel_id      = self._sym(rel_name)
            task_weight = weight_tensor[rel_id] if rel_id < len(weight_tensor) else 1.0

            if self.no_meta:
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
                query_scores = self.matcher(query, support, query_meta, support_meta)
                if self._is_medical:
                    # Preserved from medical original: false reuses query_meta
                    false_scores = self.matcher(false, support, query_meta, support_meta)
                else:
                    false_scores = self.matcher(false, support, false_meta, support_meta)

            if self._is_medical:
                # Adversarial negative weighting — medical original
                adv_temperature = 1.0
                neg_num = false_scores.size(0) // query_scores.size(0)
                if neg_num > 0:
                    f_scores = false_scores.view(query_scores.size(0), neg_num)
                    with torch.no_grad():
                        adv_weights = F.softmax(f_scores * adv_temperature, dim=-1)
                    false_scores = (adv_weights * f_scores).sum(dim=-1)

            loss = (F.relu(self.margin - (query_scores - false_scores)) * task_weight).mean()
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            if hasattr(self, 'grad_clip') and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.matcher.parameters(), self.grad_clip)
            self.optim.step()

            if self.batch_nums % self.log_every == 0:
                logging.critical(
                    f'Batch {self.batch_nums}: Loss={np.mean(losses):.4f} '
                    f'(Task: {rel_name}, Weight: {float(task_weight):.1f})'
                )

            if self.batch_nums % self.eval_every == 0 and self.batch_nums > 0:
                hits10, hits5, mrr, _ = self.eval(mode='dev', meta=self.meta)
                self.save()
                if hits10 > best_hits10:
                    self.save(self.save_path + '_bestHits10')
                    best_hits10 = hits10

            self.batch_nums += 1
            self.scheduler.step()

            if self.batch_nums >= self.max_batches:
                logging.critical(f'Max batches ({self.max_batches}) reached. Final Eval Starting.')
                hits10, hits5, mrr, _ = self.eval(mode='dev', meta=self.meta)
                logging.critical(f'FINAL DEV RESULTS - HITS@10: {hits10:.3f}, MRR: {mrr:.3f}')
                self.save()
                break

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path=None):
        path = path or self.save_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        m = self.matcher.module if isinstance(self.matcher, nn.DataParallel) else self.matcher
        torch.save(m.state_dict(), path)

    def load(self):
        state = torch.load(self.save_path, map_location=self.device)
        m = self.matcher.module if isinstance(self.matcher, nn.DataParallel) else self.matcher
        m.load_state_dict(state)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def load_tasks(self, file_path):
        if file_path.endswith('.json'):
            return json.load(open(file_path))
        tasks = defaultdict(list)
        with open(file_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h = data.get('head')     or data.get('h')
                r = data.get('relation') or data.get('r')
                t = data.get('tail')     or data.get('t')
                if not h or not t:
                    q = data.get('query') or data.get('query_enc')
                    if isinstance(q, list) and len(q) >= 2:
                        h, t = q[0], q[1]
                if h and r and t:
                    tasks[r].append([h, r, t])
        return tasks

    def _resolve_eval_file(self, mode):
        if mode == 'dev':
            primary = os.path.join(self.dataset, self.val_file)
            if os.path.exists(primary):
                return primary
            for fallback in ['validation_tasks.json', 'dev_tasks.json']:
                p = os.path.join(self.dataset, fallback)
                if os.path.exists(p):
                    logging.warning(f'val_file not found; using {p}')
                    return p
            raise FileNotFoundError(f'No dev file found in {self.dataset}')
        else:
            p = os.path.join(self.dataset, self.test_file)
            if os.path.exists(p):
                return p
            raise FileNotFoundError(f'Test file not found: {p}')

    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        hits10, hits5, hits1, mrr = [], [], [], []
        EVAL_BATCH = 1024

        semantic_relations = {
            'containedIn', 'partOfPathway', 'regulatesGO',
            'positivelyRegulatesGO', 'negativelyRegulatesGO'
        }

        task_file  = self._resolve_eval_file(mode)
        test_tasks = self.load_tasks(task_file)

        min_triples = 2 if self._is_medical else 1

        for rel_name, triples in tqdm(test_tasks.items(), desc="Evaluating Relations"):
            if len(triples) < min_triples:
                logging.warning(f'Skipping {rel_name} — too few examples ({len(triples)})')
                continue

            raw_candidates = self.rel2candidates.get(rel_name, [])
            if not raw_candidates:
                logging.warning(f'No candidates for {rel_name}; skipped.')
                continue

            support_triples = triples[:self.few]
            support_pairs   = [[self._sym(t[0]), self._sym(t[2])] for t in support_triples]
            support         = torch.LongTensor(support_pairs).to(self.device)

            if meta:
                s_l = [self._ent(t[0]) for t in support_triples]
                s_r = [self._ent(t[2]) for t in support_triples]
                support_meta = tuple(t.to(self.device) for t in self.get_meta(s_l, s_r))

            rel_h10, rel_h5, rel_h1, rel_mrr = [], [], [], []

            for triple in triples[self.few:]:
                true_ent = triple[2]
                h_esc    = self.escape_token(triple[0])
                r_esc    = self.escape_token(triple[1])

                # Filtered ranking
                valid_candidate_data = []
                for ent in raw_candidates:
                    ent_esc = self.escape_token(ent)
                    if (ent != true_ent and
                            h_esc in self.e1rel_e2 and
                            r_esc in self.e1rel_e2.get(h_esc, {}) and
                            ent_esc in self.e1rel_e2[h_esc].get(r_esc, {})):
                        continue
                    valid_candidate_data.append((
                        self._sym(triple[0]),
                        self._sym(ent),
                        self._ent(triple[0]) if meta else None,
                        self._ent(ent)        if meta else None,
                        ent
                    ))

                # Ensure true entity present — append at end (medical convention)
                if not any(d[4] == true_ent for d in valid_candidate_data):
                    valid_candidate_data.append((
                        self._sym(triple[0]), self._sym(true_ent),
                        self._ent(triple[0]) if meta else None,
                        self._ent(true_ent)  if meta else None,
                        true_ent
                    ))

                all_scores = []
                for i in range(0, len(valid_candidate_data), EVAL_BATCH):
                    batch       = valid_candidate_data[i: i + EVAL_BATCH]
                    batch_syms  = [[b[0], b[1]] for b in batch]
                    query_batch = torch.LongTensor(batch_syms).to(self.device)

                    if meta:
                        q_l    = [b[2] for b in batch]
                        q_r    = [b[3] for b in batch]
                        q_meta = tuple(t.to(self.device) for t in self.get_meta(q_l, q_r))
                        scores = self.matcher(query_batch, support, q_meta, support_meta)
                    else:
                        scores = self.matcher(query_batch, support)

                    if scores.dim() == 0:
                        all_scores.append(scores.item())
                    else:
                        all_scores.extend(scores.detach().cpu().numpy().tolist())

                if not all_scores:
                    continue

                # True score: medical appends true at end, others find by entity name
                if self._is_medical:
                    true_score = all_scores[-1]
                else:
                    true_idx   = next(i for i, d in enumerate(valid_candidate_data) if d[4] == true_ent)
                    true_score = all_scores[true_idx]

                rank = int(np.sum(np.array(all_scores) > true_score)) + 1

                hits10.append(1.0 if rank <= 10 else 0.0)
                hits5.append(1.0  if rank <= 5  else 0.0)
                hits1.append(1.0  if rank <= 1  else 0.0)
                mrr.append(1.0 / rank)
                rel_h10.append(1.0 if rank <= 10 else 0.0)
                rel_h5.append(1.0  if rank <= 5  else 0.0)
                rel_h1.append(1.0  if rank <= 1  else 0.0)
                rel_mrr.append(1.0 / rank)

            flag = '(Semantic)' if rel_name in semantic_relations else ''
            logging.critical(
                f'- {rel_name} MRR:{np.mean(rel_mrr):.3f}, '
                f'H10:{np.mean(rel_h10):.3f}, H5:{np.mean(rel_h5):.3f}, '
                f'H1:{np.mean(rel_h1):.3f} {flag}'
            )

        logging.critical(f'- Overall HITS10: {np.mean(hits10):.3f}')
        logging.critical(f'- Overall HITS5:  {np.mean(hits5):.3f}')
        logging.critical(f'- Overall HITS1:  {np.mean(hits1):.3f}')
        logging.critical(f'- Overall MRR:    {np.mean(mrr):.3f}')

        self.matcher.train()
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        self.eval(mode='dev',  meta=self.meta)
        self.eval(mode='test', meta=self.meta)


# ============================================================
if __name__ == '__main__':
    args = read_options()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    os.makedirs('./logs_', exist_ok=True)
    fh = logging.FileHandler(f'./logs_/log-{args.prefix}.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(args)
    if args.test:
        trainer.test_()
    else:
        trainer.train()
