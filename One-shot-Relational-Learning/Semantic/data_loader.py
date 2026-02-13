import json
import random
from tqdm import tqdm
import logging
from collections import defaultdict

def train_generate_simple(dataset, batch_size, few, symbol2id):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]
        train_and_test = train_tasks[query]
        random.shuffle(train_and_test)
        support_triples = train_and_test[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]  

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)
        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

        false_pairs = []
        for triple in query_triples:
            e_h = triple[0]
            e_t = triple[2]
            while True:
                noise = random.choice(candidates)
                if noise != e_t:
                    break
            false_pairs.append([symbol2id[e_h], symbol2id[noise]])

        yield support_pairs, query_pairs, false_pairs

def train_generate(dataset, batch_size, few, symbol2id, ent2id, e1rel_e2):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    def escape_token(token):
        return str(token).replace(" ", "_")

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]

        if len(candidates) <= 20:
            continue

        train_and_test = train_tasks[query]
        random.shuffle(train_and_test)
        support_triples = train_and_test[:few]

        support_pairs = [[symbol2id[escape_token(triple[0])], symbol2id[escape_token(triple[2])]] for triple in support_triples]
        support_left = [ent2id[escape_token(triple[0])] for triple in support_triples]
        support_right = [ent2id[escape_token(triple[2])] for triple in support_triples]

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) == 0:
            continue

        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)

        query_pairs = [[symbol2id[escape_token(triple[0])], symbol2id[escape_token(triple[2])]] for triple in query_triples]
        query_left = [ent2id[escape_token(triple[0])] for triple in query_triples]
        query_right = [ent2id[escape_token(triple[2])] for triple in query_triples]

        false_pairs, false_left, false_right = [], [], []
        for triple in query_triples:
            e_h = escape_token(triple[0])
            rel = escape_token(triple[1])
            e_t = escape_token(triple[2])
            key = e_h + rel
            while True:
                noise = escape_token(random.choice(candidates))
                if key in e1rel_e2 and noise in e1rel_e2[key]:
                    continue
                if noise == e_t:
                    continue
                break
            
            false_pairs.append([symbol2id[e_h], symbol2id[noise]]) 
            false_left.append(ent2id[e_h])
            false_right.append(ent2id[noise])

        yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right

def train_generate_medical(dataset, batch_size, few, symbol2id, ent2id, e1rel_e2, train_file='train_tasks.json'):
    logging.info(f'LOADING MEDICAL TRAINING DATA FROM {train_file}')
    
    file_path = dataset + '/' + train_file
    if train_file.endswith('.jsonl'):
        train_tasks = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                h = data.get('head') or data.get('h')
                r = data.get('relation') or data.get('r')
                t = data.get('tail') or data.get('t')
                if not h or not t:
                    q = data.get('query') or data.get('query_enc')
                    if isinstance(q, list) and len(q) >= 2:
                        h, t = q[0], q[1]
                if h and r and t:
                    train_tasks[r].append([h, r, t])
    else:
        train_tasks = json.load(open(file_path))

    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    def escape_token(token):
        return str(token).replace(" ", "_")

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates.get(query, [])

        if len(candidates) <= 20:
            continue

        train_and_test = train_tasks[query]
        random.shuffle(train_and_test)
        
        support_pairs, support_left, support_right = [], [], []
        query_pairs, query_left, query_right = [], [], []
        false_pairs, false_left, false_right = [], [], []

        support_triples = train_and_test[:few]
        for triple in support_triples:
            h, t = escape_token(triple[0]), escape_token(triple[2])
            support_pairs.append([symbol2id.get(h, 0), symbol2id.get(t, 0)])
            support_left.append(ent2id.get(h, 0))
            support_right.append(ent2id.get(t, 0))

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) == 0:
            continue

        query_triples = random.sample(all_test_triples, batch_size) if len(all_test_triples) >= batch_size else [random.choice(all_test_triples) for _ in range(batch_size)]

        for triple in query_triples:
            h, r, t = escape_token(triple[0]), escape_token(triple[1]), escape_token(triple[2])
            query_pairs.append([symbol2id.get(h, 0), symbol2id.get(t, 0)])
            query_left.append(ent2id.get(h, 0))
            query_right.append(ent2id.get(t, 0))

            key = h + r
            while True:
                noise = escape_token(random.choice(candidates))
                if key in e1rel_e2 and noise in e1rel_e2[key]:
                    continue
                if noise == t:
                    continue
                break
            
            false_pairs.append([symbol2id.get(h, 0), symbol2id.get(noise, 0)])
            false_left.append(ent2id.get(h, 0))
            false_right.append(ent2id.get(noise, 0))

        if len(support_pairs) > 0 and len(query_pairs) > 0:
            yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right, query
