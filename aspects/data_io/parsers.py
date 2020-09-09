import gzip
import json
import pickle
from collections import defaultdict
from typing import Set

import pandas as pd
import srsly
from more_itertools import flatten
from tqdm import tqdm

from aspects.utilities import settings


def parse_gzip(path):
    with gzip.open(path, 'rb') as gzip_file:
        for line in gzip_file:
            yield eval(line)


def amazon_dataset_to_spacy_pretrain(dataset_path):
    with open(dataset_path, 'r') as f:
        reviews = [
            {'text': text}
            for text in tqdm(json.load(f).values())
        ]
    srsly.write_jsonl(dataset_path.replace('.json', '.jsonl'), reviews)


def conceptnet_io_parse(langs: Set[str] = None):
    if langs is None:
        langs = {u'en'}
    with gzip.open(settings.CONCEPTNET_IO_PATH_GZ.as_posix(), 'rt') as conceptnet_io_file:
        conceptnet_relations = defaultdict(list)
        for line in tqdm(conceptnet_io_file):
            relation_elements = line.split('\t')
            concept_start = relation_elements[2].split('/')[3]
            concept_end = relation_elements[3].split('/')[3]
            start_lang = relation_elements[2].split('/')[2]
            end_lang = relation_elements[3].split('/')[2]
            if start_lang in langs and end_lang in langs:
                concept_relation = {
                    'start': concept_start,
                    'start-lang': start_lang,
                    'end': concept_end,
                    'end-lang': end_lang,
                    'weight': float(simplejson.loads(relation_elements[4])['weight']),
                    'relation': relation_elements[1].split('/')[2]
                }
                # it will be easier to look up by aspects
                conceptnet_relations[concept_start].append(concept_relation)
                conceptnet_relations[concept_end].append(concept_relation)
    with open(settings.CONCEPTNET_IO_PKL.as_posix(), 'wb') as conceptnet_io_file:
        pickle.dump(conceptnet_relations, conceptnet_io_file)


def conceptnet_dump_to_df():
    with open(settings.CONCEPTNET_IO_PKL.as_posix(), 'rb') as conceptnet_io_file:
        conceptnetio = pickle.load(conceptnet_io_file)
    return pd.DataFrame(list(flatten(
        [
            concepts_relations
            for concept, concepts_relations
            in conceptnetio.items()
        ]))).drop_duplicates()
