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
# --- CRITICAL: Import the Gated 1-Hop Matcher ---
from matcher_one_hop_gated import EmbedMatcher 
from tensorboardX import SummaryWriter

class Trainer(object):

    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)

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

        # Matcher setup
        self.matcher = EmbedMatcher(
            self.embed_dim,
            self.num_symbols,
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

        logging.info('BUILDING CONNECTION MATRIX (1-hop ONLY)')
        degrees = self.build_connection(max_=self.max_neighbor)
        # Note: We skipped build_2hop_connection because this model doesn't use it!

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        # Move large static tensors to GPU
        self.connections = torch.LongTensor(self.connections).to(self.device)
        e1_degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(e1_degrees_list).to(self.device)

        print(f"Trainer initialized. Using device: {self.device}, GPU count: {torch.cuda.device_count()}")

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

        if self.embed_model == 'ComplEx':
            try:
                ent_embed = np.loadtxt(ent_file)
                rel_embed = np.loadtxt(rel_file)
            except ValueError:
                ent_embed_c = np.loadtxt(ent_file, dtype=np.complex64)
                rel_embed_c = np.loadtxt(rel_file, dtype=np.complex64)
                ent_embed = np.concatenate([ent_embed_c.real, ent_embed_c.imag], axis=-1)
                rel_embed = np.concatenate([rel_embed_c.real, rel_embed_c.imag], axis=-1)
        else:
            ent_embed = np.loadtxt(ent_file)
            rel_embed = np.loadtxt(rel_file)

        if self.embed_model == 'ComplEx':
            ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
            ent_std = np.std(ent_embed, axis=1, keepdims=True)
            rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
            rel_std = np.std(rel_embed, axis=1, keepdims=True)
            eps = 1e-3
            ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
            rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

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
        embeddings.append(list(np.zeros((rel_embed.shape[1],))))
        self.symbol2id = symbol_id
        self.symbol2vec = np.array(embeddings)

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)

        def get_inverse_relation(rel):
            return rel if rel.endswith('_inv') else rel + '_inv'

        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Building 1-hop"):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                inv_rel = get_inverse_relation(rel)
                self.e1_rele2[e2].append((self.symbol2id[inv_rel], self.symbol2id[e1]))

        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2.get(ent, [])
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]
        return None

    def get_meta(self, left, right):
        left_idx = torch.LongTensor(left).to(self.device)
        right_idx = torch.LongTensor(right).to(self.device)

        left_connections_1hop = self.connections[left_idx, :, :]
        left_degrees_1hop = self.e1_degrees_tensor[left_idx]
        right_connections_1hop = self.connections[right_idx, :, :]
        right_degrees_1hop = self.e1_degrees_tensor[right_idx]

        # --- TRICK: Pass None for 2nd Hop ---
        # The matcher expects 6 items but will ignore items 2 and 5
        return (left_connections_1hop, None, left_degrees_1hop,
                right_connections_1hop, None, right_degrees_1hop)

    def train(self):
        logging.info('START TRAINING...')
        best_hits10 = 0.0
        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2):
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            support_meta = tuple(t.to(self.device, non_blocking=True) if t is not None else None for t in self.get_meta(support_left, support_right))
            query_meta = tuple(t.to(self.device, non_blocking=True) if t is not None else None for t in self.get_meta(query_left, query_right))
            false_meta = tuple(t.to(self.device, non_blocking=True) if t is not None else None for t in self.get_meta(false_left, false_right))

            support = torch.LongTensor(support).pin_memory().to(self.device, non_blocking=True)
            query = torch.LongTensor(query).pin_memory().to(self.device, non_blocking=True)
            false = torch.LongTensor(false).pin_memory().to(self.device, non_blocking=True)

            query_scores = self.matcher(query, support, query_meta, support_meta)
            false_scores = self.matcher(false, support, false_meta, support_meta)

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

    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        # Modified line to use validation_tasks.json for dev mode
        test_tasks = json.load(open(self.dataset + ('/validation_tasks.json' if mode == 'dev' else '/test_tasks.json')))
        rel2candidates = self.rel2candidates
        hits10, hits5, hits1, mrr = [], [], [], []

        EVAL_BATCH_SIZE = 1024
        SAMPLE_SIZE = 1000 

        for query_ in tqdm(test_tasks.keys(), desc="Evaluating Relations", mininterval=10.0):
            hits10_, hits5_, hits1_, mrr_ = [], [], [], []
            all_candidates = rel2candidates[query_]
            
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[self.escape_token(triple[0])],
                              symbol2id[self.escape_token(triple[2])]] for triple in support_triples]

            if meta:
                support_left = [self.ent2id[self.escape_token(triple[0])] for triple in support_triples]
                support_right = [self.ent2id[self.escape_token(triple[2])] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)
                support_meta = tuple(t.to(self.device, non_blocking=True) if t is not None else None for t in support_meta)

            support = torch.LongTensor(support_pairs).to(self.device)

            for triple in tqdm(test_tasks[query_][few:], desc=f"Tasks in {query_}", leave=False, mininterval=30.0):
                true = triple[2]
                h_esc = self.escape_token(triple[0])
                h_sym = symbol2id[h_esc]
                h_ent = self.ent2id[h_esc] if meta else None

                # Sampled Evaluation
                current_candidates = [true] 
                if len(all_candidates) > SAMPLE_SIZE:
                    negs = random.sample(all_candidates, SAMPLE_SIZE)
                    for n in negs:
                        if n != true:
                            current_candidates.append(n)
                    current_candidates = current_candidates[:SAMPLE_SIZE+1]
                else:
                    current_candidates = all_candidates

                valid_candidate_data = []
                for ent in current_candidates:
                    ent_esc = self.escape_token(ent)
                    t_sym = symbol2id[ent_esc]
                    t_ent = self.ent2id[ent_esc] if meta else None
                    valid_candidate_data.append((h_sym, t_sym, h_ent, t_ent))

                all_scores = []
                for i in range(0, len(valid_candidate_data), EVAL_BATCH_SIZE):
                    batch = valid_candidate_data[i : i + EVAL_BATCH_SIZE]
                    batch_syms = [[b[0], b[1]] for b in batch]
                    query_batch = torch.LongTensor(batch_syms).to(self.device)
                    
                    if meta:
                        batch_left = [b[2] for b in batch]
                        batch_right = [b[3] for b in batch]
                        query_meta = tuple(t.to(self.device, non_blocking=True) if t is not None else None for t in self.get_meta(batch_left, batch_right))
                        scores_t = self.matcher(query_batch, support, query_meta, support_meta)
                    else:
                        scores_t = self.matcher(query_batch, support)
                    all_scores.extend(scores_t.detach().cpu().numpy())

                true_score = all_scores[0]
                all_scores_np = np.array(all_scores)
                rank = np.sum(all_scores_np > true_score) + 1
                
                hits10.append(1.0 if rank <= 10 else 0.0)
                hits5.append(1.0 if rank <= 5 else 0.0)
                hits1.append(1.0 if rank <= 1 else 0.0)
                mrr.append(1.0 / rank)

            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(
                query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
            
        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MAP: {:.3f}'.format(np.mean(mrr)))

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
