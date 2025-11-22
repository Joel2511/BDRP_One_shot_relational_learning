# trainer_atomic.py (MODIFIED FOR DEBUGGING AND ATOMIC STABILITY)
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
from data_loader import * # Assuming this contains train_generate
from matcher import * # Assuming this contains EmbedMatcher
from tensorboardX import SummaryWriter

class Trainer(object):
    
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        # Force GPU usage
        if not torch.cuda.is_available():
            # NOTE: Removed the RuntimeError for debug runs, but keep this check
            self.device = torch.device("cpu")
            logging.warning("No CUDA found. Running on CPU, expect extreme slowness.")
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
            self.load_embed() # <--- CALLS THE EMBEDDING LOADER
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

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))  
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        # Move large static tensors to GPU
        self.connections = torch.LongTensor(self.connections).to(self.device)
        e1_degrees_list = [float(self.e1_degrees.get(i, 0)) for i in range(self.num_ents)]
        self.e1_degrees_tensor = torch.FloatTensor(e1_degrees_list).to(self.device)

        print(f"Trainer initialized. Using device: {self.device}, GPU count: {torch.cuda.device_count()}")

    # --- SYMBOL / EMBEDDING LOADERS (COMPLLEX FIX APPLIED) ---
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

        if self.embed_model in ['DistMult', 'TransE', 'RESCAL','TransH']:
            # Normal loading for purely real-valued models
            ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)
        
        elif self.embed_model == 'ComplEx':
            # --- START OF COMPLLEX FIX ---
            ent_file = self.dataset + '/entity2vec.' + self.embed_model
            rel_file = self.dataset + '/relation2vec.' + self.embed_model
            
            try:
                # 1. Attempt to load the data specifying the complex data type (dytpe=np.complex64).
                ent_embed_complex = np.loadtxt(ent_file, dtype=np.complex64)
                rel_embed_complex = np.loadtxt(rel_file, dtype=np.complex64)
                
                # 2. Flatten the complex array into a real-valued array (Real | Imaginary)
                ent_embed = np.concatenate([ent_embed_complex.real, ent_embed_complex.imag], axis=-1)
                rel_embed = np.concatenate([rel_embed_complex.real, rel_embed_complex.imag], axis=-1)
                
                logging.info('ComplEx embeddings loaded and flattened successfully.')

            except ValueError as e:
                # Fallback to standard float loading if complex format fails (assuming manual pre-flattening)
                if 'to float' in str(e) or 'complex' in str(e):
                    logging.warning(f"Complex loading failed (Error: {e}). Trying to load as standard floats.")
                    ent_embed = np.loadtxt(ent_file)
                    rel_embed = np.loadtxt(rel_file)
                else:
                    raise e
            # --- END OF COMPLLEX FIX ---
        
        else:
            raise ValueError(f"Unknown embed_model: {self.embed_model}")
        
        # --- ComplEx Normalization (Applied only if ComplEx was the chosen model) ---
        if self.embed_model == 'ComplEx':
            ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
            ent_std = np.std(ent_embed, axis=1, keepdims=True)
            rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
            rel_std = np.std(rel_embed, axis=1, keepdims=True)
            eps = 1e-3
            ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
            rel_embed = (rel_embed - rel_mean) / (rel_std + eps)
        
        # --- Common Symbol Mapping Logic ---
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
        # Ensure padding vector has the correct flattened dimension (2*D)
        embeddings.append(list(np.zeros((rel_embed.shape[1],)))) 
        self.symbol2id = symbol_id
        self.symbol2vec = np.array(embeddings)

    # --- CONNECTION MATRIX ---
    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)

        def get_inverse_relation(rel):
            return rel if rel.endswith('_inv') else rel + '_inv'

        # Use tqdm here for progress update on the most time-consuming step before training
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Building Connection Matrix"): 
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

    # --- TRAINING LOOP (LOGGING AND EVAL FIXES APPLIED) ---
    def train(self):
        logging.info('START TRAINING...')
        best_hits10 = 0.0
        losses = deque([], self.log_every)
        margins = deque([], self.log_every)
        
        logging.info('LOADING TRAINING DATA (This may be the slow step)')

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2):
            
            # --- BATCH PREPARATION ---
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            support_meta = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(support_left, support_right))
            query_meta  = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(query_left, query_right))
            false_meta  = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(false_left, false_right))

            support = torch.LongTensor(support).pin_memory().to(self.device, non_blocking=True)
            query   = torch.LongTensor(query).pin_memory().to(self.device, non_blocking=True)
            false   = torch.LongTensor(false).pin_memory().to(self.device, non_blocking=True)

            # --- FORWARD PASS ---
            if self.no_meta:
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
                query_scores = self.matcher(query, support, query_meta, support_meta)
                false_scores = self.matcher(false, support, false_meta, support_meta)

            margin_ = query_scores - false_scores
            margins.append(margin_.mean().item())
            loss = F.relu(self.margin - margin_).mean()
            
            # --- BACKWARD PASS ---
            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # --- LOGGING FIX (Will show output every batch if log_every=1) ---
            if self.batch_nums % self.log_every == 0:
                avg_loss = np.mean(losses)
                avg_margin = np.mean(margins)
                logging.critical(f"Batch {self.batch_nums}: Loss={avg_loss:.4f}, Margin={avg_margin:.4f}")
                if self.writer:
                    self.writer.add_scalar('Avg_batch_loss', avg_loss, self.batch_nums)
                    self.writer.add_scalar('MARGIN', avg_margin, self.batch_nums)
                losses = deque([], self.log_every)
                margins = deque([], self.log_every)


            if self.batch_nums % self.eval_every == 0 and self.batch_nums > 0:
                hits10, hits5, mrr = self.eval(meta=self.meta)
                if self.writer:
                    self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                    self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                    self.writer.add_scalar('MAP', mrr, self.batch_nums)

                self.save()
                if hits10 > best_hits10:
                    self.save(self.save_path + '_bestHits10')
                    best_hits10 = hits10

            self.batch_nums += 1
            self.scheduler.step()
            if self.batch_nums >= self.max_batches: # Use >= for safety
                logging.critical(f"Max batches ({self.max_batches}) reached. Saving final model.")
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

    # --- EVALUATION (FILE PATH FIX APPLIED) ---
    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        # CRITICAL FIX: Changed /validation_tasks.json to /dev_tasks.json
        test_tasks = json.load(open(self.dataset + ('/dev_tasks.json' if mode=='dev' else '/test_tasks.json')))
        rel2candidates = self.rel2candidates
        hits10, hits5, hits1, mrr = [], [], [], []

        for query_ in test_tasks.keys():
            hits10_, hits5_, hits1_, mrr_ = [], [], [], []
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[self.escape_token(triple[0])],
                              symbol2id[self.escape_token(triple[2])]] for triple in support_triples]

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
                    e0, er, ent_esc = self.escape_token(triple[0]), self.escape_token(triple[1]), self.escape_token(ent)
                    if (e0 not in self.e1rel_e2 or
                        er not in self.e1rel_e2[e0] or
                        ent_esc not in self.e1rel_e2[e0][er]) and ent != true:
                        query_pairs.append([symbol2id[ent_esc], symbol2id[e0]])
                        if meta:
                            query_left.append(self.ent2id[e0])
                            query_right.append(self.ent2id[ent_esc])

                query = torch.LongTensor(query_pairs).to(self.device, non_blocking=True)
                if meta:
                    query_meta = tuple(t.to(self.device, non_blocking=True) for t in self.get_meta(query_left, query_right))
                    scores_t = self.matcher(query, support, query_meta, support_meta)
                else:
                    scores_t = self.matcher(query, support)

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
