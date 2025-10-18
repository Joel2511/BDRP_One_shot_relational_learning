from itertools import chain

import pandas as pd
from datasets import load_dataset


def get_stats():
    relation = []
    entity = []
    size = []
    data = load_dataset("relbert/nell")
    splits = data.keys()
    for split in splits:
        df = data[split].to_pandas()
        size.append({
            "number of pairs": len(df),
            "number of unique relation types": len(df["relation"].unique())
        })
        relation.append(df.groupby('relation')['head'].count().to_dict())
        entity += [df.groupby('head_type')['head'].count().to_dict(), df.groupby('tail_type')['tail'].count().to_dict()]
    relation = pd.DataFrame(relation, index=[f"number of pairs ({s})" for s in splits]).T
    relation = relation.fillna(0).astype(int)
    entity = pd.DataFrame(entity, index=list(chain(*[[f"head ({s})", f"tail ({s})"] for s in splits]))).T
    entity = entity.fillna(0).astype(int)
    size = pd.DataFrame(size, index=splits).T
    return relation, entity, size

df_relation, df_entity, df_size = get_stats()
print(f"\n- Number of instances\n\n {df_size.to_markdown()}")
print(f"\n- Number of pairs in each relation type\n\n {df_relation.to_markdown()}")
print(f"\n- Number of entity types\n\n {df_entity.to_markdown()}")

