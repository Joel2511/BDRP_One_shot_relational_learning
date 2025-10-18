import json
import datasets


logger = datasets.logging.get_logger(__name__)
_DESCRIPTION = """Few shots link prediction dataset. """
_NAME = "nell"
_VERSION = "0.0.8"
_CITATION = """
@inproceedings{xiong-etal-2018-one,
    title = "One-Shot Relational Learning for Knowledge Graphs",
    author = "Xiong, Wenhan  and
      Yu, Mo  and
      Chang, Shiyu  and
      Guo, Xiaoxiao  and
      Wang, William Yang",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1223",
    doi = "10.18653/v1/D18-1223",
    pages = "1980--1990",
    abstract = "Knowledge graphs (KG) are the key components of various natural language processing applications. To further expand KGs{'} coverage, previous studies on knowledge graph completion usually require a large number of positive examples for each relation. However, we observe long-tail relations are actually more common in KGs and those newly added relations often do not have many known triples for training. In this work, we aim at predicting new facts under a challenging setting where only one training instance is available. We propose a one-shot relational learning framework, which utilizes the knowledge distilled by embedding models and learns a matching metric by considering both the learned embeddings and one-hop graph structures. Empirically, our model yields considerable performance improvements over existing embedding models, and also eliminates the need of re-training the embedding models when dealing with newly added relations.",
}
"""

_HOME_PAGE = "https://github.com/asahi417/relbert"
_URL = f'https://huggingface.co/datasets/relbert/{_NAME}/resolve/main/data'
_TYPE = "nell_filter"
_URLS = {
    str(datasets.Split.TRAIN): [f'{_URL}/{_TYPE}.train.jsonl'],
    str(datasets.Split.VALIDATION): [f'{_URL}/{_TYPE}.validation.jsonl'],
    str(datasets.Split.TEST): [f'{_URL}/{_TYPE}.test.jsonl']
}


class NELLConfig(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NELLConfig, self).__init__(**kwargs)


class NELL(datasets.GeneratorBasedBuilder):
    """Dataset."""

    BUILDER_CONFIGS = [NELLConfig(name=_TYPE, version=datasets.Version(_VERSION), description=_DESCRIPTION)]

    def _split_generators(self, dl_manager):
        # downloaded_file = dl_manager.download_and_extract(_URLS[self.config.name])
        downloaded_file = dl_manager.download_and_extract(_URLS)
        return [datasets.SplitGenerator(name=i, gen_kwargs={"filepaths": downloaded_file[str(i)]})
                for i in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]]

    def _generate_examples(self, filepaths):
        _key = 0
        for filepath in filepaths:
            logger.info(f"generating examples from = {filepath}")
            with open(filepath, encoding="utf-8") as f:
                _list = [i for i in f.read().split('\n') if len(i) > 0]
                for i in _list:
                    data = json.loads(i)
                    yield _key, data
                    _key += 1

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "relation": datasets.Value("string"),
                    "head": datasets.Value("string"),
                    "head_type": datasets.Value("string"),
                    "tail": datasets.Value("string"),
                    "tail_type": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
        )