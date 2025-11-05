# trainer1_fixed.py
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
from matcher import *
from tensorboardX import SummaryWriter

class Trainer(object):
    
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        # device: prefer GPUs if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # improve cuDNN perf for fixed-size inputs
        if torch.cuda.is_available():
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

        # If symbol2vec exists as numpy array, keep it (matcher likely handles numpy)
        # but ensure it's in a consistent dtype if we need to pass tensor later.
        # Create matcher instance
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

        # DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            self.matcher = torch.nn.DataParallel(self.matcher)

        # move matcher to device
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

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json')) 

        # load answer dict
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        # Move large static arrays to device (torch tensors)
        # connections currently is numpy array shaped (num_ents, max_, 2)
        self.connections = torch.LongTensor(self.connections).to(self.device)
        # e1_degrees is a dict keyed by ent id; build contiguous tensor in ent-id order
        e1_degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(e1_degrees_list).to(self.device)


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
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)

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
                if key not in ['','OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key],:]))
            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key],:]))
            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings  # keep as numpy for matcher init (unchanged behavior)


    def build_connection(self, max_=100):
        # initialize with pad id
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)

        def get_inverse_relation(rel):
            return rel if rel.endswith('_inv') else rel + '_inv'

        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                # guard: ensure rel and ent keys exist in symbol2id
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


    def get_meta(self, left, right):
        # left, right may be Python lists of ids. Convert to LongTensor on device to index.
        if isinstance(left, (list, tuple, np.ndarray)):
            left_idx = torch.LongTensor(left).to(self.device)
        else:
            left_idx = left.to(self.device)
        if isinstance(right, (list, tuple, np.ndarray)):
            right_idx = torch.LongTensor(right).to(self.device)
        else:
            right_idx = right.to(self.device)

        left_connections = self.connections[left_idx, :, :]      # already on device
        left_degrees = self.e1_degrees_tensor[left_idx]
        right_connections = self.connections[right_idx, :, :]
        right_degrees = self.e1_degrees_tensor[right_idx]
        return (left_connections, left_degrees, right_connections, right_degrees)


    def train(self):
        logging.info('START TRAINING...')
        best_hits10 = 0.0
        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2):
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            # move batch tensors to device
            support = torch.LongTensor(support).to(self.device)
            query = torch.LongTensor(query).to(self.device)
            false = torch.LongTensor(false).to(self.device)

            if self.no_meta:
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
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

            if self.batch_nums % self.eval_every == 0:
                hits10, hits5, mrr = self.eval(meta=self.meta)
                if self.writer:
                    self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                    self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                    self.writer.add_scalar('MAP', mrr, self.batch_nums)

                self.save()
                if hits10 > best_hits10:
                    self.save(self.save_path + '_bestHits10')
                    best_hits10 = hits10

            if self.batch_nums % self.log_every == 0:
                if self.writer:
                    self.writer.add_scalar('Avg_batch_loss', np.mean(losses), self.batch_nums)

            self.batch_nums += 1
            self.scheduler.step()
            if self.batch_nums == self.max_batches:
                self.save()
                break


    def save(self, path="models/initial"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # handle DataParallel wrapper
        if isinstance(self.matcher, torch.nn.DataParallel):
            torch.save(self.matcher.module.state_dict(), path)
        else:
            torch.save(self.matcher.state_dict(), path)


    def load(self):
        # load into CPU first then to device (safer)
        state = torch.load(self.save_path, map_location=self.device)
        if isinstance(self.matcher, torch.nn.DataParallel):
            self.matcher.module.load_state_dict(state)
        else:
            self.matcher.load_state_dict(state)


    def escape_token(self, token):
        return token.replace(" ", "_")


    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/validation_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))

        rel2candidates = self.rel2candidates
        hits10, hits5, hits1, mrr = [], [], [], []

        for query_ in test_tasks.keys():
            hits10_, hits5_, hits1_, mrr_ = [], [], [], []
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [
                [symbol2id[self.escape_token(triple[0])], symbol2id[self.escape_token(triple[2])]]
                for triple in support_triples
            ]

            if meta:
                support_left = [self.ent2id[self.escape_token(triple[0])] for triple in support_triples]
                support_right = [self.ent2id[self.escape_token(triple[2])] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

            support = torch.LongTensor(support_pairs).to(self.device)

            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_pairs = [[symbol2id[self.escape_token(triple[0])], symbol2id[self.escape_token(triple[2])]]]

                if meta:
                    query_left = [self.ent2id[self.escape_token(triple[0])]]
                    query_right = [self.ent2id[self.escape_token(triple[2])]]

                for ent in candidates:
                    e0 = self.escape_token(triple[0])
                    er = self.escape_token(triple[1])
                    ent_esc = self.escape_token(ent)
                    if (e0 not in self.e1rel_e2 or
                        er not in self.e1rel_e2[e0] or
                        ent_esc not in self.e1rel_e2[e0][er]) and ent != true:
                        query_pairs.append([symbol2id[ent_esc], symbol2id[e0]])
                        if meta:
                            query_left.append(self.ent2id[e0])
                            query_right.append(self.ent2id[ent_esc])

                query = torch.LongTensor(query_pairs).to(self.device)
                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    scores_t = self.matcher(query, support, query_meta, support_meta)
                else:
                    scores_t = self.matcher(query, support)

                # move to cpu for numpy operations
                scores = scores_t.detach().cpu().numpy()
                sort = list(np.argsort(scores))[::-1]
                rank = sort.index(0) + 1
                hits10.append(1.0 if rank <= 10 else 0.0)
                hits5.append(1.0 if rank <= 5 else 0.0)
                hits1.append(1.0 if rank <= 1 else 0.0)
                mrr.append(1.0 / rank)
                hits10_.append(1.0 if rank <= 10 else 0.0)
                hits5_.append(1.0 if rank <= 5 else 0.0)
                hits1_.append(1.0 if rank <= 1 else 0.0)
                mrr_.append(1.0 / rank)

            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
            logging.info('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)
    if args.test:
        trainer.test_()
    else:
        trainer.train()
