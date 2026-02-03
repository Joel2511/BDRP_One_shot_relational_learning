import json
import logging
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict, deque
from torch import optim
from tqdm import tqdm
import os
import random

from args import read_options
from data_loader import *
from matcher_sap import * #mean + distance neighbor encoding 
from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            logging.warning("No CUDA found. Running on CPU.")
        else:
            self.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

        self.meta = not self.no_meta
        use_pretrain = not self.random_embed

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
        else:
            self.load_embed()
        self.use_pretrain = use_pretrain

        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols

        self.matcher = EmbedMatcher(
            self.embed_dim,
            self.num_symbols,
            use_pretrain=self.use_pretrain,
            embed=self.symbol2vec,
            dropout=self.dropout,
            batch_size=self.batch_size,
            process_steps=self.process_steps,
            finetune=self.fine_tune,
            aggregate=self.aggregate,
            knn_k=self.knn_k,
            knn_alpha=self.knn_alpha
        )
        if torch.cuda.device_count() > 1:
            self.matcher = torch.nn.DataParallel(self.matcher)
        self.matcher.to(self.device)

        self.batch_nums = 0
        self.writer = None if self.test else SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.5)
        self.use_semantic = args.use_semantic

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json')) 
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        # Set specific filenames for filtered data
        if self.object_only:
            self.train_file = 'medical_object_only.train.jsonl'
            self.val_file = 'medical_object_only.validation.jsonl'
            self.test_file = 'medical_object_only.test.jsonl'
        else:
            self.train_file = 'medical_cleaned.train.jsonl'
            self.val_file = 'validation_tasks.json'
            self.test_file = 'test_tasks.json'

        self.connections = torch.LongTensor(self.connections).to(self.device)
        e1_degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(e1_degrees_list).to(self.device)

        print(f"Trainer initialized. Using device: {self.device}, GPU count: {torch.cuda.device_count()}")

    
    def load_tasks(self, file_path):
            """Groups JSONL triples by relation for evaluation compatibility."""
            if file_path.endswith('.json'):
                return json.load(open(file_path))
            
            tasks = defaultdict(list)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Apply same fallback logic here
                    h = data.get('head') or data.get('h')
                    r = data.get('relation') or data.get('r')
                    t = data.get('tail') or data.get('t')
                    
                    if not h or not t:
                        q = data.get('query') or data.get('query_enc')
                        if isinstance(q, list) and len(q) >= 2:
                            h, t = q[0], q[1]
                    
                    if h and r and t:
                        tasks[r].append([h, r, t])
            return tasks

    def load_symbol2id(self):
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['','OOV']:
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
    symbol_id = {}
    rel2id = json.load(open(self.dataset + '/relation2ids'))
    ent2id = json.load(open(self.dataset + '/ent2ids'))

    logging.info('LOADING PRE-TRAINED EMBEDDING')
    ent_file = self.dataset + '/entity2vec.' + self.embed_model
    rel_file = self.dataset + '/relation2vec.' + self.embed_model

    # (Keep your existing ComplEx loading logic here...)
    ent_embed = np.loadtxt(ent_file) 
    rel_embed = np.loadtxt(rel_file)

    # 1. Standardize Structural Channel
    ent_embed = (ent_embed - np.mean(ent_embed)) / (np.std(ent_embed) + 1e-3)
    rel_embed = (rel_embed - np.mean(rel_embed)) / (np.std(rel_embed) + 1e-3)

    # 2. Integrate Semantic Channel (SUBSTITUTED FOR PUBMEDBERT)
    if hasattr(self, 'use_semantic') and self.use_semantic:
        # Determine filename based on args
        sb_filename = f'medical_{self.semantic_type}_anchors.npy'
        sb_path = os.path.join(os.path.dirname(self.dataset), sb_filename)
        
        if os.path.exists(sb_path):
            logging.info(f'INTEGRATING {self.semantic_type.upper()} SEMANTIC CHANNEL')
            sb = np.load(sb_path)

            # Standardize independently
            sb = (sb - sb.mean(axis=0, keepdims=True)) / (sb.std(axis=0, keepdims=True) + 1e-3)

            # Project 768D to 100D to match structural (Maintains 200D total)
            if sb.shape[1] != ent_embed.shape[1]:
                rng = np.random.RandomState(42)
                proj = rng.normal(0, 1.0 / np.sqrt(sb.shape[1]), size=(sb.shape[1], ent_embed.shape[1]))
                sb = sb @ proj

            semantic_weight = 0.1 
            ent_embed = np.concatenate([ent_embed, semantic_weight * sb], axis=1)

            rel_padding = np.zeros((rel_embed.shape[0], sb.shape[1]))
            rel_embed = np.concatenate([rel_embed, rel_padding], axis=1)

            logging.info(f'Final embedding dim: {ent_embed.shape[1]}')
        else:
            logging.error(f'Semantic file not found at {sb_path}')
    
        # Final symbol mapping (Cleaned and Deduplicated)
        embeddings = []
        i = 0
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(rel_embed[rel2id[key], :]))
    
        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(ent_embed[ent2id[key], :]))
    
        symbol_id['PAD'] = i
        embeddings.append(list(np.zeros((ent_embed.shape[1],))))
    
        self.symbol2id = symbol_id
        self.symbol2vec = np.array(embeddings)


      
    # --- CONNECTION MATRIX ---
    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)

        def get_inverse_relation(rel):
            return rel if rel.endswith('_inv') else rel + '_inv'

        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Building Connections"):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                inv_rel = get_inverse_relation(rel)
                self.e1_rele2[e2].append((self.symbol2id[inv_rel], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2.get(ent, [])
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]
        return degrees

    # --- META EXTRACTION ---
    def get_meta(self, left, right):
        left_idx = torch.LongTensor(left).to(self.device)
        right_idx = torch.LongTensor(right).to(self.device)

        left_connections = self.connections[left_idx,:,:]
        left_degrees = self.e1_degrees_tensor[left_idx]
        right_connections = self.connections[right_idx,:,:]
        right_degrees = self.e1_degrees_tensor[right_idx]
        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')
        best_hits10 = 0.0
        losses = deque([], self.log_every)
        
        # ----- SMART RELATION PENALTY WEIGHTS -----
        # Use inverse sqrt of train frequency for weighting
        relation_freq = {}  # will store counts before training
        train_path = os.path.join(self.dataset, self.train_file)
        with open(train_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                rel = data.get('relation') or data.get('r')
                if rel:
                    relation_freq[rel] = relation_freq.get(rel, 0) + 1
        
        # Find max freq to normalize weights
        max_freq = max(relation_freq.values()) if relation_freq else 1
        epsilon = 1e-6
        
        weight_tensor = torch.ones(len(self.symbol2id)).to(self.device)
        
        for rel, freq in relation_freq.items():
            esc = self.escape_token(rel)
            if esc in self.symbol2id:
                rid = self.symbol2id[esc]
                # inverse sqrt weighting
                w = (max_freq / (freq + epsilon)) ** 0.5
                # clamp to avoid too huge
                w = float(min(max(w, 1.0), 10.0))
                weight_tensor[rid] = w
                logging.info(f"Relation {rel} freq={freq}, weight={w:.3f}")

        
        from data_loader import train_generate_medical 
        
        # Pass the specific train_file (Object-Only or Full)
        for data in train_generate_medical(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2, train_file=self.train_file):
            if len(data) != 10:
                raise ValueError(f"CRITICAL ERROR: Data loader yielded {len(data)} items instead of 10.")
    
            support_p, query_p, false_p, s_l, s_r, q_l, q_r, f_l, f_r, rel_name = data
    
            support_meta = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(s_l, s_r))
            query_meta   = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(q_l, q_r))
            false_meta   = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(f_l, f_r))
    
            support = torch.clamp(torch.LongTensor(support_p).to(self.device), 0, self.pad_id)
            query   = torch.clamp(torch.LongTensor(query_p).to(self.device), 0, self.pad_id)
            false   = torch.clamp(torch.LongTensor(false_p).to(self.device), 0, self.pad_id)
            
            esc_rel_name = self.escape_token(rel_name)
            rel_id = self.symbol2id.get(esc_rel_name, self.pad_id)
            task_weight = weight_tensor[rel_id] if rel_id < len(weight_tensor) else 1.0
    
            if self.no_meta:
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
                query_scores = self.matcher(query, support, query_meta, support_meta)
                false_scores = self.matcher(false, support, false_meta, support_meta)
    
            adv_temperature = 1.0
            neg_num = false_scores.size(0) // query_scores.size(0)
            f_scores = false_scores.view(query_scores.size(0), neg_num)
            
            with torch.no_grad():
                adv_weights = F.softmax(f_scores * adv_temperature, dim=-1)
            
            adv_false_scores = (adv_weights * f_scores).sum(dim=-1)
            
            loss = (F.relu(self.margin - (query_scores - adv_false_scores)) * task_weight).mean()
            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
            if self.batch_nums % self.log_every == 0:
                avg_loss = np.mean(losses)
                logging.critical(f"Batch {self.batch_nums}: Loss={avg_loss:.4f} (Task: {rel_name}, Weight: {task_weight:.1f})")
    
            if self.batch_nums % self.eval_every == 0 and self.batch_nums > 0:
                hits10, hits5, mrr, _ = self.eval(mode='dev', meta=self.meta)
                self.save()
                if hits10 > best_hits10:
                    self.save(self.save_path + '_bestHits10')
                    best_hits10 = hits10
    
            self.batch_nums += 1
            self.scheduler.step()
    
            if self.batch_nums >= self.max_batches:
                logging.critical(f"Max batches ({self.max_batches}) reached. Final Eval Starting.")
                hits10, hits5, mrr, _ = self.eval(mode='dev', meta=self.meta)
                logging.critical(f"FINAL RESULTS - HITS@10: {hits10:.3f}, MRR: {mrr:.3f}")
                self.save()
                break
                
    # --- SAVE / LOAD ---
    def save(self, path="models/initial"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(self.matcher, torch.nn.DataParallel):
            torch.save(self.matcher.module.state_dict(), path)
        else:
            torch.save(self.matcher.state_dict(), path)

    def load(self):
        state = torch.load(self.save_path, map_location=self.device)
        if isinstance(self.matcher, torch.nn.DataParallel):
            self.matcher.module.load_state_dict(state)
        else:
            self.matcher.load_state_dict(state)

    # --- OPTIMIZED EVALUATION (BATCHED for Whole Dataset) ---
    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        symbol2id = self.symbol2id
        few = self.few
    
        logging.info('EVALUATING ON %s DATA' % mode.upper())
        # Use helper to load either JSON or JSONL tasks
        task_path = os.path.join(self.dataset, self.val_file if mode == 'dev' else self.test_file)
        test_tasks = self.load_tasks(task_path)
        
        rel2candidates = self.rel2candidates
        hits10, hits5, hits1, mrr = [], [], [], []
    
        relation_metrics = {}
        semantic_relations = ["containedIn", "partOfPathway", "regulatesGO",
                              "positivelyRegulatesGO", "negativelyRegulatesGO"]
    
        EVAL_BATCH_SIZE = 1024 
    
        for query_ in tqdm(test_tasks.keys(), desc="Evaluating Relations"):
            if len(test_tasks[query_]) < 2:
                logging.warning(f"Skipping {query_} eval due to too few examples ({len(test_tasks[query_])})")
                continue

            hits10_, hits5_, hits1_, mrr_ = [], [], [], []
            candidates = rel2candidates.get(query_, [])
            if not candidates:
                continue

            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[self.escape_token(triple[0])],
                              symbol2id[self.escape_token(triple[2])]] for triple in support_triples]
    
            if meta:
                support_left = [self.ent2id[self.escape_token(triple[0])] for triple in support_triples]
                support_right = [self.ent2id[self.escape_token(triple[2])] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)
                support_meta = tuple(t.to(self.device, non_blocking=True) for t in support_meta)
    
            support = torch.LongTensor(support_pairs).to(self.device)
    
            for triple in tqdm(test_tasks[query_][few:], desc=f"Tasks in {query_}", leave=False):
                true = triple[2]
                h_esc = self.escape_token(triple[0])
                r_esc = self.escape_token(triple[1])
                h_sym = symbol2id[h_esc]
                h_ent = self.ent2id[h_esc] if meta else None
    
                valid_candidate_data = [] 
                for ent in candidates:
                    ent_esc = self.escape_token(ent)
                    if (h_esc in self.e1rel_e2 and 
                        r_esc in self.e1rel_e2[h_esc] and 
                        ent_esc in self.e1rel_e2[h_esc][r_esc]) and ent != true:
                        continue
                    t_sym = symbol2id[ent_esc]
                    t_ent = self.ent2id[ent_esc] if meta else None
                    valid_candidate_data.append((h_sym, t_sym, h_ent, t_ent))
    
                true_esc = self.escape_token(true)
                true_sym = symbol2id[true_esc]
                true_ent = self.ent2id[true_esc] if meta else None
                valid_candidate_data.append((h_sym, true_sym, h_ent, true_ent))
                
                all_scores = []
                for i in range(0, len(valid_candidate_data), EVAL_BATCH_SIZE):
                    batch = valid_candidate_data[i : i + EVAL_BATCH_SIZE]
                    batch_syms = [[b[0], b[1]] for b in batch]
                    query_batch = torch.LongTensor(batch_syms).to(self.device)
                    
                    if meta:
                        batch_left = [b[2] for b in batch]
                        batch_right = [b[3] for b in batch]
                        query_meta = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(batch_left, batch_right))
                        scores_t = self.matcher(query_batch, support, query_meta, support_meta)
                    else:
                        scores_t = self.matcher(query_batch, support)
                    
                    if scores_t.dim() == 0:
                        all_scores.append(scores_t.item())
                    else:
                        all_scores.extend(scores_t.detach().cpu().numpy())
                
                if not all_scores:
                    all_scores = [0.0]
                
                true_score = all_scores[-1]
                all_scores_np = np.array(all_scores)
                rank = np.sum(all_scores_np > true_score) + 1
    
                hits10.append(1.0 if rank <= 10 else 0.0)
                hits5.append(1.0 if rank <= 5 else 0.0)
                hits1.append(1.0 if rank <= 1 else 0.0)
                mrr.append(1.0 / rank)
                hits10_.append(1.0 if rank <= 10 else 0.0)
                hits5_.append(1.0 if rank <= 5 else 0.0)
                hits1_.append(1.0 if rank <= 1 else 0.0)
                mrr_.append(1.0 / rank)
    
            relation_metrics[query_] = {
                "MRR": np.mean(mrr_) if mrr_ else 0,
                "HITS@10": np.mean(hits10_) if hits10_ else 0,
                "HITS@5": np.mean(hits5_) if hits5_ else 0,
                "HITS@1": np.mean(hits1_) if hits1_ else 0
            }
    
            flag = "(Semantic)" if query_ in semantic_relations else ""
            logging.critical('{} MRR:{:.3f}, H10:{:.3f}, H5:{:.3f}, H1:{:.3f} {}'.format(
                query_, np.mean(mrr_), np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), flag))
            
        logging.critical('Overall HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('Overall HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('Overall HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('Overall MRR: {:.3f}'.format(np.mean(mrr)))
    
        self.matcher.train()
        return np.mean(hits10), np.mean(hits5), np.mean(mrr), relation_metrics


    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        self.eval(mode='dev', meta=self.meta)
        self.eval(mode='test', meta=self.meta)

    def escape_token(self, token):
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(token))


if __name__ == '__main__':
    args = read_options()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    os.makedirs('./logs_', exist_ok=True)
    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

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
