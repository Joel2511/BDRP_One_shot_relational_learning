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
from matcher_nell_one import EmbedMatcher
from tensorboardX import SummaryWriter

class Trainer(object):
    
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        # Force GPU usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logging.warning("No CUDA found. Running on CPU.")
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

        # Load semantic embeddings for NELL-ONE
        self.semantic_vec = None
        if 'nell-one' in self.dataset.lower():
            semantic_path = '/gpfs/workdir/anilj/nell_data/nell_one_semantic_anchors.npy'
            logging.info(f'LOADING SEMANTIC EMBEDDINGS from {semantic_path}')
            raw_sem = torch.from_numpy(np.load(semantic_path)).float().to(self.device)
            # FIX: L2 Normalize so it operates on the same scale as ComplEx
            import torch.nn.functional as F
            self.semantic_vec = F.normalize(raw_sem, p=2, dim=1)

        # Matcher setup
        semantic_dim = self.semantic_vec.shape[1] if self.semantic_vec is not None else 0
        self.matcher = EmbedMatcher(
            self.embed_dim,
            self.num_symbols,
            semantic_dim=semantic_dim,
            use_pretrain=self.use_pretrain,
            embed=self.symbol2vec,
            dropout=self.dropout,
            batch_size=self.batch_size,
            process_steps=self.process_steps,
            finetune=self.fine_tune,
            aggregate=self.aggregate
        )
        if torch.cuda.device_count() > 1:
            self.matcher = torch.nn.DataParallel(self.matcher)
        self.matcher.to(self.device)

        self.batch_nums = 0
        self.writer = None if self.test else SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATE ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json')) 
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        # Move large static tensors to GPU
        self.connections = torch.LongTensor(self.connections).to(self.device)
        e1_degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(e1_degrees_list).to(self.device)

        logging.info(f"Trainer initialized. Device: {self.device}, GPU count: {torch.cuda.device_count()}")

    # --- UNIVERSAL SAFE LOADING ---
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
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDINGS')
        
        ent_file = self.dataset + '/entity2vec.' + self.embed_model
        rel_file = self.dataset + '/relation2vec.' + self.embed_model

        if self.embed_model in ['DistMult', 'TransE', 'RESCAL','TransH']:
            ent_embed = np.loadtxt(ent_file)
            rel_embed = np.loadtxt(rel_file)
        
        elif self.embed_model == 'ComplEx':
            try:
                ent_embed = np.loadtxt(ent_file)
                rel_embed = np.loadtxt(rel_file)
                logging.info("Loaded ComplEx as standard floats.")
            except ValueError:
                logging.info("Flattening complex strings...")
                ent_embed_c = np.loadtxt(ent_file, dtype=np.complex64)
                rel_embed_c = np.loadtxt(rel_file, dtype=np.complex64)
                ent_embed = np.concatenate([ent_embed_c.real, ent_embed_c.imag], axis=-1)
                rel_embed = np.concatenate([rel_embed_c.real, rel_embed_c.imag], axis=-1)

        if self.embed_model == 'ComplEx':
            eps = 1e-3
            ent_embed = (ent_embed - np.mean(ent_embed, axis=1, keepdims=True)) / (np.std(ent_embed, axis=1, keepdims=True) + eps)
            rel_embed = (rel_embed - np.mean(rel_embed, axis=1, keepdims=True)) / (np.std(rel_embed, axis=1, keepdims=True) + eps)

        embeddings = []
        i = 0
        for key in rel2id.keys():
            if key not in ['','OOV']:
                embeddings.append(list(rel_embed[rel2id[key],:]))
                i += 1
        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                embeddings.append(list(ent_embed[ent2id[key],:]))
                i += 1
        embeddings.append(list(np.zeros((ent_embed.shape[1],))))
        self.symbol2id = {**rel2id, **ent2id, 'PAD': i}
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

    # --- TRAINING LOOP ---
    def train(self):
        logging.info('START TRAINING...')
        best_hits10 = 0.0
        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2):
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            support_meta = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(support_left, support_right))
            query_meta   = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(query_left, query_right))
            false_meta   = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(false_left, false_right))

            support = torch.LongTensor(support).pin_memory().to(self.device, non_blocking=True)
            query   = torch.LongTensor(query).pin_memory().to(self.device, non_blocking=True)
            false   = torch.LongTensor(false).pin_memory().to(self.device, non_blocking=True)

            query_sem = self.semantic_vec
            support_sem = self.semantic_vec

            if self.no_meta:
                query_scores = self.matcher(query, support, query_sem=query_sem, support_sem=support_sem)
                false_scores = self.matcher(false, support, query_sem=query_sem, support_sem=support_sem)
            else:
                query_scores = self.matcher(query, support, query_meta=query_meta, support_meta=support_meta,
                                            query_sem=query_sem, support_sem=support_sem)
                false_scores = self.matcher(false, support, query_meta=false_meta, support_meta=support_meta,
                                            query_sem=query_sem, support_sem=support_sem)

            margin_ = query_scores - false_scores
            margins.append(margin_.mean().item())
            loss = F.relu(self.margin - margin_).mean()
            
            if self.writer:
                self.writer.add_scalar('MARGIN', np.mean(margins), self.batch_nums)

            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.batch_nums % self.log_every == 0:
                avg_loss = np.mean(losses)
                logging.critical(f"Batch {self.batch_nums}: Loss={avg_loss:.4f}")
                if self.writer:
                    self.writer.add_scalar('Avg_batch_loss', avg_loss, self.batch_nums)

            if self.batch_nums % self.eval_every == 0 and self.batch_nums > 0:
                hits10, hits5, mrr = self.eval(meta=self.meta)
                if self.writer:
                    self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                    self.writer.add_scalar('MAP', mrr, self.batch_nums)
                self.save()
                if hits10 > best_hits10:
                    self.save(self.save_path + '_bestHits10')
                    best_hits10 = hits10

            self.batch_nums += 1
            self.scheduler.step()
            
            if self.batch_nums >= self.max_batches:
                logging.critical(f"Max batches ({self.max_batches}) reached. Running final evaluation.")
                hits10, hits5, mrr = self.eval(meta=self.meta)
                logging.critical(f"FINAL DEV RESULTS - HITS@10: {hits10:.3f}, MRR: {mrr:.3f}")
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

    # --- EVALUATION ---
    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            dev_path_std = os.path.join(self.dataset, 'validation_tasks.json')
            dev_path_alt = os.path.join(self.dataset, 'dev_tasks.json')
            if os.path.exists(dev_path_std):
                test_tasks_all = json.load(open(dev_path_std))
            elif os.path.exists(dev_path_alt):
                test_tasks_all = json.load(open(dev_path_alt))
            else:
                raise FileNotFoundError(f"No dev tasks file found in {self.dataset}")
        else:
            test_tasks_all = json.load(open(os.path.join(self.dataset, 'test_tasks.json')))

        rel2candidates = self.rel2candidates
        hits10, hits5, mrr = [], [], []
        
        # New List to store per-relation results
        relation_results = {}

        for query_ in tqdm(test_tasks_all.keys(), desc="Evaluating Relations"):
            candidates = rel2candidates[query_]
            support_triples = test_tasks_all[query_][:few]
            support_pairs = [[symbol2id[self.escape_token(triple[0])],
                              symbol2id[self.escape_token(triple[2])]] for triple in support_triples]

            if meta:
                # (Support Meta Code - unchanged)
                support_left = [self.ent2id[self.escape_token(triple[0])] for triple in support_triples]
                support_right = [self.ent2id[self.escape_token(triple[2])] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)
                support_meta = tuple(t.to(self.device, non_blocking=True) for t in support_meta)
            else:
                support_meta = None

            support = torch.LongTensor(support_pairs).to(self.device)
            
            # Metric storage for THIS specific relation
            rel_hits10 = []
            rel_hits5 = []
            rel_mrr = []

            for triple in test_tasks_all[query_][few:]:
                true = triple[2]
                h_sym = symbol2id[self.escape_token(triple[0])]
                t_sym = symbol2id[self.escape_token(true)]
                
                # --- CANDIDATE GENERATION (Use the fix from before!) ---
                query_pairs = []
                left_ents = []
                right_ents = []

                for cand in candidates:
                    cand_esc = self.escape_token(cand)
                    if cand_esc in symbol2id and cand_esc in self.ent2id:
                        query_pairs.append([h_sym, symbol2id[cand_esc]])
                        left_ents.append(self.ent2id[self.escape_token(triple[0])])
                        right_ents.append(self.ent2id[cand_esc])
                
                query_pairs.append([h_sym, t_sym])
                left_ents.append(self.ent2id[self.escape_token(triple[0])])
                right_ents.append(self.ent2id[self.escape_token(true)])

                query_batch = torch.LongTensor(query_pairs).to(self.device)

                if meta:
                    query_meta_single = self.get_meta(left_ents, right_ents) # Using List fix
                    query_meta_single = tuple(t.to(self.device, non_blocking=True) for t in query_meta_single)
                else:
                    query_meta_single = None

                query_sem = self.semantic_vec
                support_sem = self.semantic_vec

                scores = self.matcher(query_batch, support, query_meta=query_meta_single, support_meta=support_meta,
                                      query_sem=query_sem, support_sem=support_sem)

                rank = torch.sum(scores > scores[-1]).item() + 1
                
                # Update Global Metrics
                hits10.append(1.0 if rank <= 10 else 0.0)
                hits5.append(1.0 if rank <= 5 else 0.0)
                mrr.append(1.0 / rank)
                
                # Update Per-Relation Metrics
                rel_hits10.append(1.0 if rank <= 10 else 0.0)
                rel_hits5.append(1.0 if rank <= 5 else 0.0)
                rel_mrr.append(1.0 / rank)

            # Store average for this relation
            relation_results[query_] = {
                'hits10': np.mean(rel_hits10),
                'hits5': np.mean(rel_hits5),
                'mrr': np.mean(rel_mrr)
            }

        # --- LOG THE PER-RELATION BREAKDOWN ---
        logging.critical("-" * 40)
        logging.critical("PER-RELATION BREAKDOWN:")
        for rel, res in relation_results.items():
            logging.critical(f"Relation: {rel} | Hits@10: {res['hits10']:.3f} | MRR: {res['mrr']:.3f}")
        logging.critical("-" * 40)

        logging.critical('OVERALL HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('OVERALL HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('OVERALL MAP: {:.3f}'.format(np.mean(mrr)))

        self.matcher.train()
        return np.mean(hits10), np.mean(hits5), np.mean(mrr)

    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        self.eval(mode='dev', meta=self.meta)
        self.eval(mode='test', meta=self.meta)

    def escape_token(self, token):
        return token.replace(" ", "_")


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
