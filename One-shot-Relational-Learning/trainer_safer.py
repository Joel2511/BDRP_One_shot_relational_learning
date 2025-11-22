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
from data_loader_safer import train_generate_safer # SAFER-specific loader
from matcher_safer import SAFERMatcher      # NEW: use the SAFER matcher
from tensorboardX import SummaryWriter
from grapher_safer import Graph                   # Needed for extracting subgraphs for SAFER

class Trainer(object):

    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. This trainer requires a GPU.")
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

        # SAFER matcher initialization
        self.matcher = SAFERMatcher(
            self.embed_dim,
            self.num_symbols,
            use_pretrain=self.use_pretrain,
            embed=self.symbol2vec,
            dropout=self.dropout
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

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        # SAFER: instantiate Graph for extracting subgraphs
        self.grapher = Graph(self.dataset) 

        print(f"SAFER Trainer initialized. Using device: {self.device}, GPU count: {torch.cuda.device_count()}")

    # --- SYMBOL / EMBEDDING LOADERS ---
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
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL','TransH']:
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
            self.symbol2id = symbol_id
            self.symbol2vec = np.array(embeddings)

    # --- TRAINING LOOP ---
    def train(self):
        logging.info('START TRAINING...')
        best_hits10 = 0.0
        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        safer_gen = train_generate_safer(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2, self.grapher)

        for data in safer_gen:
            support_subgraphs, query_subgraphs, false_subgraphs = data

            support_subgraphs = [ss for ss in support_subgraphs if len(ss) > 0]
            query_subgraphs = [qs for qs in query_subgraphs if len(qs) > 0]
            false_subgraphs = [fs for fs in false_subgraphs if len(fs) > 0]

            if len(support_subgraphs) == 0 or len(query_subgraphs) == 0 or len(false_subgraphs) == 0:
                continue

            # SAFER matcher: cosine similarities for positive and negative (margin loss)
            pos_scores = self.matcher(support_subgraphs, query_subgraphs)
            neg_repr = self.matcher.score_negatives(support_subgraphs, false_subgraphs)
            pos_scores = pos_scores[:len(neg_repr)]  # align batch
            # For simplicity: margin loss as in original trainer
            margin_ = pos_scores - F.cosine_similarity(neg_repr, pos_scores.unsqueeze(1))
            margins.append(margin_.mean().item())
            loss = F.relu(self.margin - margin_).mean()

            if self.writer:
                self.writer.add_scalar('MARGIN', np.mean(margins), self.batch_nums)

            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.batch_nums % self.eval_every == 0:
                hits10, hits5, mrr = self.eval()
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
    def eval(self, mode='dev'):
        self.matcher.eval()
        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        test_tasks = json.load(open(self.dataset + ('/validation_tasks.json' if mode=='dev' else '/test_tasks.json')))
        rel2candidates = self.rel2candidates
        hits10, hits5, hits1, mrr = [], [], [], []

        for query_ in test_tasks.keys():
            hits10_, hits5_, hits1_, mrr_ = [], [], [], []
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]

            # Create support/query subgraphs for each candidate
            support_subgraphs = [self.grapher.pair_feature((self.escape_token(triple[0]), self.escape_token(triple[2]))) for triple in support_triples]
            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_subgraphs = [self.grapher.pair_feature((self.escape_token(triple[0]), self.escape_token(triple[2])))]
                candidate_scores = []
                for ent in candidates:
                    # Form new subgraph for each candidate tail
                    candidate_subgraph = self.grapher.pair_feature((self.escape_token(triple[0]), self.escape_token(ent)))
                    score = self.matcher(support_subgraphs, [candidate_subgraph]).item()
                    candidate_scores.append(score)
                # Compute rank
                sort = list(np.argsort(candidate_scores))[::-1]
                rank = sort.index(candidates.index(true)) + 1
                hits10.append(1.0 if rank <= 10 else 0.0)
                hits5.append(1.0 if rank <= 5 else 0.0)
                hits1.append(1.0 if rank <= 1 else 0.0)
                mrr.append(1.0 / rank)
                hits10_.append(1.0 if rank <= 10 else 0.0)
                hits5_.append(1.0 if rank <= 5 else 0.0)
                hits1_.append(1.0 if rank <= 1 else 0.0)
                mrr_.append(1.0 / rank)

            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(
                query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
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
        self.eval(mode='dev')
        self.eval(mode='test')

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
