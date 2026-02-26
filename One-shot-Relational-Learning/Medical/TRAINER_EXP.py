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
from MATCHER import EmbedMatcher

# ============================================================
#  DATASET CONFIG REGISTRY
#  Each prefix maps to its unique hyperparameters and flags.
#  Add ATOMIC here once its data is ready.
# ============================================================
DATASET_CONFIGS = {
    # ---- NELL-One ----
    # Sparse graph (avg degree 6.57). LUKE embeddings give genuine semantic signal.
    # step_size=1000 acts as regularization (model peaks early around batch 2000-3000).
    # max pooling, tight LR decay.
    'nell': {
        'step_size':    1000,
        'gate_mode':    'learned',    # full learned sigmoid gate
        'margin':       1.0,
        'weight_cap':   10.0,
        'train_file':   'train_tasks.json',
        'val_file':     'validation_tasks.json',
        'test_file':    'test_tasks.json',
        'is_medical':   False,
        'object_only':  False,
        'escape_keys':  False,
    },
    # ---- FB15k-237 ----
    # Dense graph (avg degree 85.31). BERT gives noise on Freebase path strings.
    # Gate frozen to fixed blend: 90% structural, 10% semantic.
    # Looser LR decay (step_size=3000) because model needs longer to converge.
    'fb15k': {
        'step_size':    3000,
        'gate_mode':    'fixed',      # fixed alpha blend, no learned gate
        'gate_alpha':   0.9,          # structural weight
        'margin':       5.0,
        'weight_cap':   10.0,
        'train_file':   'train_tasks.json',
        'val_file':     'validation_tasks.json',
        'test_file':    'test_tasks.json',
        'is_medical':   False,
        'object_only':  False,
        'escape_keys':  False,
    },
    # ---- Medical / OntoOmics ----
    # Hierarchical graph (avg degree 63.62). PubMedBERT embeddings.
    # Uses object_only filter. escape_keys=True for ontology URI formatting.
    # Weight cap lowered to 5.0 to prevent rare relations dominating gradients.
    'medical': {
        'step_size':    3000,
        'gate_mode':    'learned',
        'margin':       3.0,
        'weight_cap':   5.0,
        'train_file':   'medical_object_only.train.jsonl',
        'val_file':     'medical_object_only.validation.jsonl',
        'test_file':    'medical_object_only.test.jsonl',
        'is_medical':   True,
        'object_only':  True,
        'escape_keys':  True,
    },
    # ---- ATOMIC ----
    # Extreme sparse commonsense graph (avg degree 8.36, 82% sparsity).
    # Slot reserved — fill once ATOMIC data/embeddings are ready.
    'atomic': {
        'step_size':    1000,
        'gate_mode':    'learned',
        'margin':       1.0,
        'weight_cap':   10.0,
        'train_file':   'train_tasks.json',
        'val_file':     'validation_tasks.json',
        'test_file':    'test_tasks.json',
        'is_medical':   False,
        'object_only':  False,
        'escape_keys':  False,
    },
}

def get_dataset_config(prefix):
    """
    Match a prefix string to a config block.
    E.g. 'nell_one_semantic_luke' -> 'nell' config.
    Falls back to 'nell' defaults if no match found.
    """
    prefix_lower = prefix.lower()
    for key in DATASET_CONFIGS:
        if key in prefix_lower:
            return DATASET_CONFIGS[key]
    logging.warning(f"No config found for prefix '{prefix}'. Defaulting to nell config.")
    return DATASET_CONFIGS['nell']


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        # ---- device ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logging.warning("No CUDA found. Running on CPU.")
        torch.backends.cudnn.benchmark = True

        # ---- dataset config ----
        self.cfg = get_dataset_config(self.prefix)
        logging.info(f"Dataset config loaded for prefix '{self.prefix}': {self.cfg}")

        self.meta       = not self.no_meta
        self.use_pretrain = not self.random_embed

        # ---- file names (config overrides args if not explicitly set) ----
        # Allow CLI overrides; fall back to config defaults
        if not getattr(self, 'train_file', None):
            self.train_file = self.cfg['train_file']
        if not getattr(self, 'val_file', None):
            self.val_file   = self.cfg['val_file']
        if not getattr(self, 'test_file', None):
            self.test_file  = self.cfg['test_file']

        # ---- load entity/relation maps ----
        self.ent2id        = json.load(open(os.path.join(self.dataset, 'ent2ids')))
        self.num_ents      = len(self.ent2id)
        self.rel2candidates = json.load(open(os.path.join(self.dataset, 'rel2candidates.json')))
        self.e1rel_e2      = json.load(open(os.path.join(self.dataset, 'e1rel_e2.json')))

        # ---- load symbols + embeddings ----
        logging.info("LOADING SYMBOL ID AND SYMBOL EMBEDDING")
        if self.test or self.random_embed:
            self.load_symbol2id()
            self.use_pretrain = False
        else:
            self.load_embed()

        self.num_symbols = len(self.symbol2id) - 1
        self.pad_id      = self.num_symbols

        # ---- build matcher ----
        self.matcher = EmbedMatcher(
            embed_dim     = self.embed_dim,
            num_symbols   = self.num_symbols,
            use_pretrain  = self.use_pretrain,
            embed         = self.symbol2vec,
            dropout       = self.dropout,
            batch_size    = self.batch_size,
            process_steps = self.process_steps,
            finetune      = self.fine_tune,
            aggregate     = self.aggregate,
            knn_k         = self.knn_k,
            gate_mode     = self.cfg['gate_mode'],
            gate_alpha    = self.cfg.get('gate_alpha', None),
        )
        if torch.cuda.device_count() > 1:
            self.matcher = nn.DataParallel(self.matcher)
        self.matcher.to(self.device)

        # ---- optimizer: lower LR for gate in learned mode ----
        m = self.matcher.module if isinstance(self.matcher, nn.DataParallel) else self.matcher
        if self.cfg['gate_mode'] == 'learned':
            gate_params = list(m.gate_layer.parameters())
            base_params = [p for n, p in m.named_parameters() if 'gate_layer' not in n]
            self.optim = optim.Adam([
                {'params': base_params, 'lr': self.lr},
                {'params': gate_params, 'lr': self.lr * 0.1},
            ], weight_decay=self.weight_decay)
        else:
            # Fixed gate: no gate params to train separately
            self.optim = optim.Adam(
                m.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # ---- LR scheduler: dataset-specific step_size ----
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optim, step_size=self.cfg['step_size'], gamma=0.5
        )

        # ---- connection matrix ----
        self.max_neighbor = min(self.max_neighbor, self.num_ents)
        self.knn_k        = min(self.knn_k, self.max_neighbor)
        self.connections  = None
        self.e1_degrees_tensor = None
        self.build_connection_matrix(max_=self.max_neighbor)

        # ---- misc ----
        self.batch_nums = 0
        self.writer     = None if self.test else SummaryWriter('logs/' + self.prefix)
        print(f"Trainer initialized on {self.device} | config: {self.cfg}")

    # ------------------------------------------------------------------
    #  Token escaping (needed for medical ontology URIs)
    # ------------------------------------------------------------------
    def escape_token(self, token):
        if self.cfg['escape_keys']:
            return re.sub(r'[^a-zA-Z0-9_]', '_', str(token))
        return str(token)

    def _sym(self, token):
        """Look up symbol2id with optional escaping."""
        return self.symbol2id.get(self.escape_token(token), self.pad_id)

    def _ent(self, token):
        """Look up ent2id with optional escaping."""
        return self.ent2id.get(self.escape_token(token), 0)

    # ------------------------------------------------------------------
    #  Symbol / Embedding loading
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
        rel2id = json.load(open(os.path.join(self.dataset, 'relation2ids')))
        ent2id = json.load(open(os.path.join(self.dataset, 'ent2ids')))

        logging.info("LOADING PRE-TRAINED EMBEDDING")
        ent_embed = np.loadtxt(os.path.join(self.dataset, f'entity2vec.{self.embed_model}'))
        rel_embed = np.loadtxt(os.path.join(self.dataset, f'relation2vec.{self.embed_model}'))

        # standardise
        ent_embed = (ent_embed - ent_embed.mean()) / (ent_embed.std() + 1e-3)
        rel_embed = (rel_embed - rel_embed.mean()) / (rel_embed.std() + 1e-3)
        logging.info(f"Loaded {self.embed_model} as standard floats.")

        # ---- semantic channel ----
        if getattr(self, 'use_semantic', False):
            # Medical uses a named file; others use prefix_anchors.npy
            if self.cfg['is_medical']:
                sb_file = os.path.join(
                    os.path.dirname(self.dataset),
                    f'medical_{self.semantic_type}_anchors.npy'
                )
            else:
                sb_file = os.path.join(
                    os.path.dirname(self.dataset),
                    f'{self.prefix}_anchors.npy'
                )

            if os.path.exists(sb_file):
                logging.info(f"Integrating semantic channel from {sb_file}")
                sb = np.load(sb_file)
                sb = (sb - sb.mean(axis=0, keepdims=True)) / (sb.std(axis=0, keepdims=True) + 1e-3)

                if sb.shape[1] != ent_embed.shape[1]:
                    rng  = np.random.RandomState(42)
                    proj = rng.normal(0, 1.0 / np.sqrt(sb.shape[1]),
                                      size=(sb.shape[1], ent_embed.shape[1]))
                    sb = sb @ proj

                ent_embed = np.concatenate([ent_embed, sb], axis=1)
                rel_embed = np.concatenate([rel_embed,
                                            np.zeros((rel_embed.shape[0], sb.shape[1]))], axis=1)
                logging.info(f"Unified embedding dim: {ent_embed.shape[1]}")
            else:
                logging.warning(f"Semantic .npy not found at {sb_file}")

        # ---- build symbol table ----
        symbol_id  = {}
        embeddings = []
        i = 0

        for key in rel2id:
            if key not in ['', 'OOV']:
                symbol_id[self.escape_token(key)] = i
                i += 1
                embeddings.append(rel_embed[rel2id[key]].tolist())

        for key in ent2id:
            if key not in ['', 'OOV']:
                symbol_id[self.escape_token(key)] = i
                i += 1
                embeddings.append(ent_embed[ent2id[key]].tolist())

        symbol_id['PAD'] = i
        embeddings.append(np.zeros(ent_embed.shape[1]).tolist())

        self.symbol2id  = symbol_id
        self.symbol2vec = np.array(embeddings)

    # ------------------------------------------------------------------
    #  Connection matrix
    # ------------------------------------------------------------------
    def build_connection_matrix(self, max_=100):
        """
        Build the neighbor connection matrix.
        FIX: shuffle before truncation to remove file-order bias.
        Dense graphs (FB15k, Medical) hit the cap frequently; without shuffling
        the same relation types are always selected, missing the tail types.
        """
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
            neighbors = self.e1_rele2.get(ent, [])
            if len(neighbors) > max_:
                random.shuffle(neighbors)
            neighbors = neighbors[:max_]
            self.e1_degrees[idx] = len(neighbors)
            for j, (rid, eid) in enumerate(neighbors):
                self.connections[idx, j, 0] = rid
                self.connections[idx, j, 1] = eid

        self.connections = torch.LongTensor(self.connections).to(self.device)
        degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(degrees_list).to(self.device)

    # ------------------------------------------------------------------
    #  Meta helpers
    # ------------------------------------------------------------------
    def get_meta(self, left, right):
        l = torch.LongTensor(left).to(self.device)
        r = torch.LongTensor(right).to(self.device)
        return (self.connections[l], self.e1_degrees_tensor[l],
                self.connections[r], self.e1_degrees_tensor[r])

    # ------------------------------------------------------------------
    #  Task weights
    # ------------------------------------------------------------------
    def compute_task_weights(self):
        relation_freq = defaultdict(int)
        with open(os.path.join(self.dataset, self.train_file), 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    rel  = data.get('relation') or data.get('r')
                    if rel:
                        relation_freq[rel] += 1
                except json.JSONDecodeError:
                    continue

        max_freq   = max(relation_freq.values()) if relation_freq else 1
        weight_cap = self.cfg['weight_cap']
        weight_tensor = torch.ones(len(self.symbol2id), device=self.device)
        for rel, freq in relation_freq.items():
            rid = self._sym(rel)
            if rid < len(weight_tensor):
                w = float(min(max((max_freq / (freq + 1e-6)) ** 0.5, 1.0), weight_cap))
                weight_tensor[rid] = w
        return weight_tensor

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def train(self):
        logging.info("START TRAINING...")
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
            # FIX: false triples use their own neighborhood, not query's
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
                false_scores = self.matcher(false, support, false_meta, support_meta)

            loss = (F.relu(self.margin - (query_scores - false_scores)) * task_weight).mean()
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.batch_nums % self.log_every == 0:
                logging.critical(
                    f"Batch {self.batch_nums}: Loss={np.mean(losses):.4f} "
                    f"(Task {rel_name}, Weight {float(task_weight):.1f})"
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
                logging.info("EVALUATING ON DEV DATA")
                hits10, hits5, mrr, _ = self.eval(mode='dev', meta=self.meta)
                logging.critical(f"FINAL DEV RESULTS - HITS@10: {hits10:.3f}, MRR: {mrr:.3f}")
                self.save()
                break

    # ------------------------------------------------------------------
    #  Save / Load
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
    #  Evaluation
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
                h = data.get('head')   or data.get('h')
                r = data.get('relation') or data.get('r')
                t = data.get('tail')   or data.get('t')
                if not h or not t:
                    q = data.get('query') or data.get('query_enc')
                    if isinstance(q, list) and len(q) >= 2:
                        h, t = q[0], q[1]
                if h and r and t:
                    tasks[r].append([h, r, t])
        return tasks

    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        hits10, hits5, hits1, mrr = [], [], [], []
        EVAL_BATCH = 1024

        task_file = os.path.join(
            self.dataset, self.val_file if mode == 'dev' else self.test_file
        )
        test_tasks = self.load_tasks(task_file)

        for rel_name, triples in test_tasks.items():
            if len(triples) < 1:
                logging.warning(f"No triples for {rel_name}; skipped.")
                continue

            raw_candidates = self.rel2candidates.get(rel_name, [])
            if not raw_candidates:
                logging.warning(f"No candidates for {rel_name}; skipped.")
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
                true_ent  = triple[2]
                h_esc     = self.escape_token(triple[0])
                r_esc     = self.escape_token(triple[1])

                # --- build valid candidate set ---
                valid_cands = []
                for ent in raw_candidates:
                    ent_esc = self.escape_token(ent)
                    # filter already-known true answers (except the target)
                    if (ent != true_ent and
                            h_esc in self.e1rel_e2 and
                            r_esc in self.e1rel_e2.get(h_esc, {}) and
                            ent_esc in self.e1rel_e2[h_esc].get(r_esc, {})):
                        continue
                    valid_cands.append(ent)

                if true_ent not in valid_cands:
                    valid_cands.append(true_ent)

                # --- score in batches ---
                all_scores = []
                for i in range(0, len(valid_cands), EVAL_BATCH):
                    batch_cands  = valid_cands[i: i + EVAL_BATCH]
                    batch_syms   = [[self._sym(triple[0]), self._sym(c)] for c in batch_cands]
                    query_batch  = torch.LongTensor(batch_syms).to(self.device)

                    if meta:
                        q_l = [self._ent(triple[0])] * len(batch_cands)
                        q_r = [self._ent(c) for c in batch_cands]
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

                true_idx   = valid_cands.index(true_ent)
                true_score = all_scores[true_idx]
                rank       = int(np.sum(np.array(all_scores) > true_score)) + 1

                h10 = 1.0 if rank <= 10 else 0.0
                h5  = 1.0 if rank <= 5  else 0.0
                h1  = 1.0 if rank <= 1  else 0.0
                mr  = 1.0 / rank

                hits10.append(h10); hits5.append(h5)
                hits1.append(h1);   mrr.append(mr)
                rel_h10.append(h10); rel_h5.append(h5)
                rel_h1.append(h1);   rel_mrr.append(mr)

            logging.critical(
                f"- {rel_name} Hits10:{np.mean(rel_h10):.3f}, "
                f"Hits5:{np.mean(rel_h5):.3f}, Hits1:{np.mean(rel_h1):.3f} "
                f"MRR:{np.mean(rel_mrr):.3f}"
            )

        logging.critical(f"- HITS10: {np.mean(hits10):.3f}")
        logging.critical(f"- HITS5:  {np.mean(hits5):.3f}")
        logging.critical(f"- HITS1:  {np.mean(hits1):.3f}")
        logging.critical(f"- MAP:    {np.mean(mrr):.3f}")

        self.matcher.train()
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

    def test_(self):
        self.load()
        logging.info("Pre-trained model loaded")
        self.eval(mode='dev',  meta=self.meta)
        self.eval(mode='test', meta=self.meta)


# ============================================================
if __name__ == "__main__":
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
