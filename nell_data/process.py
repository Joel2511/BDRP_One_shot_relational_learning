"""
- Wiki-One https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz
- NELL-One https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz

wget https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz
tar -xzf nell.tar.gz

wget https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz
tar -xzf wiki.tar.gz

"""
import os
import json
import re
from itertools import chain

data_dir_nell = "NELL"
os.makedirs("data", exist_ok=True)

short = ['alcs', "uk", "us", "usa", "npr", "nbc", "bbc", "cnn", "abc", "cbs", "nfl", "mlb", "nba", "nhl", "pga", "ncaa",
         "wjhu", "pbs", "un"]
non_entity_types = [
    'academicfield',
    'agent',
    'agriculturalproduct',
    'amphibian',
    'animal',
    'aquarium',
    'arachnid',
    'architect',
    'arthropod',
    'bakedgood',
    'bathroomitem',
    'bedroomitem',
    'beverage',
    'bird',
    'blog',
    'bodypart',
    'bone',
    'candy',
    'cave',
    'chemical',
    'clothing',
    'coffeedrink',
    'condiment',
    'crimeorcharge',
    'crustacean',
    'date',
    'dateliteral',
    'economicsector',
    'fish',
    'food',
    'fruit',
    'fungus',
    'furniture',
    'grain',
    'hallwayitem',
    'hobby',
    'insect',
    'invertebrate',
    'jobposition',
    'kitchenitem',
    'landscapefeatures',
    'legume',
    'location',
    'mammal',
    'meat',
    'mlsoftware',
    'mollusk',
    'month',
    'nut',
    'officebuildingroom',
    'physiologicalcondition',
    'plant',
    'politicsissue',
    'profession',
    'professionalorganization',
    'reptile',
    'room',
    'sport',
    'tableitem',
    'tradeunion',
    'vegetable',
    'vehicle',
    'vertebrate',
    'weapon',
    'wine'
]


def clean(token):
    _, _type, token = token.split(":")
    token = token.replace("_", " ")
    token = token.replace("__", "")
    token = re.sub(r"00\d\Z", "", token)
    token = re.sub(r"\An(\d+)", r"\1", token)
    if _type in non_entity_types:
        return token, _type
    new_token = []
    for _t in token.split(" "):
        if len(_t) == 0:
            continue
        if _t in short:
            _t = _t.upper()
        else:
            _t = _t.capitalize()
        new_token.append(_t)
    return " ".join(new_token), _type


if not os.path.exists(data_dir_nell):
    raise ValueError("Please download the dataset first\n"
                     "wget https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz\n"
                     "tar -xzf nell.tar.gz")


def read_file(_file):
    with open(_file, 'r') as f_reader:
        tmp = json.load(f_reader)
    flatten = list(chain(*[[{"relation": r, "head": h, "tail": t} for (h, r, t) in v] for v in tmp.values()]))
    return flatten


def read_vocab(_file):
    with open(_file) as f_reader:
        ent2ids = json.load(f_reader)
    return sorted(list(ent2ids.keys()))


if __name__ == '__main__':
    # Process raw data
    vocab = read_vocab(f"{data_dir_nell}/ent2ids")
    vocab = [clean(i) for i in vocab if len(i.split(":")) > 2]
    vocab = ["\t".join(i) for i in vocab if len(i[0]) > 0 and len(i[1]) > 0]
    with open("data/nell.vocab.txt", 'w') as f:
        f.write("\n".join(vocab))
    vocab_term = [i.split('\t')[0] for i in vocab]

    for i, s in zip(['dev_tasks.json', 'test_tasks.json', 'train_tasks.json'], ['validation', 'test', 'train']):
        d = read_file(f"{data_dir_nell}/{i}")

        for _d in d:
            head = _d.pop("head")
            tail = _d.pop("tail")

            head_entity, head_type = clean(head)
            _d['head'] = head_entity
            _d['head_type'] = head_type
            assert head_entity in vocab_term, head_entity

            tail_entity, tail_type = clean(tail)
            _d['tail'] = tail_entity
            _d['tail_type'] = tail_type
            assert tail_entity in vocab_term, tail_entity

        with open(f"data/nell.{s}.jsonl", "w") as f:
            f.write("\n".join([json.dumps(_d) for _d in d]))

    # Filter entity relation
    full_data = {}
    for s in ["train", "validation", "test"]:
        with open(f"data/nell.{s}.jsonl") as f:
            data = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
            data = [i for i in data if i['head_type'] not in non_entity_types and i['tail_type'] not in non_entity_types]
        with open(f"data/nell_filter.{s}.jsonl", "w") as f:
            f.write('\n'.join([json.dumps(i) for i in data]))

    with open("data/nell.vocab.txt") as f:
        vocab = [i.split("\t") for i in f.read().split('\n')]

    vocab = ["\t".join([a, b]) for a, b in vocab if b not in non_entity_types]
    with open("data/nell_filter.vocab.txt", 'w') as f:
        f.write('\n'.join(vocab))

