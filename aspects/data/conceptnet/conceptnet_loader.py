import logging
from pathlib import Path
from typing import Generator

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger()

# CONCEPTNET_RELATIONS = {"IsA", "RelatedTo", "Antonym", "Synonym"}
CONCEPTNET_LANGS = ['en', 'pl']

# only for logging
CONCEPTNET_LINES_ALL = 34074917


def parse_conceptnet_dump(concepnet_path: Path) -> Generator:
    with concepnet_path.open("r") as f:
        for line in tqdm(f, total=34074917):
            relation, node_from, node_to, lang = parse_line(line)
            if relation and node_from and node_to:
                yield relation, node_from, node_to, lang


def parse_line(line_text: str):
    _, relation_text, node_from_text, node_to_text, _ = line_text.split("\t")
    node_from, lang = parse_conceptnet_node(node_from_text)
    if not node_from:
        return None, None, None, None
    node_to, lang = parse_conceptnet_node(node_to_text)
    if not node_to:
        return None, None, None, None
    relation = parse_conceptnet_relation(relation_text)
    if not relation:
        return None, None, None, None
    return relation, node_from, node_to, lang


def parse_conceptnet_node(node_text: str):
    node_text_parts = node_text.split("/")[1:]
    if node_text_parts[0] != 'c':
        # logger.error(f"node {node_text} starts with other than c prefix - misconception about data structure")
        return None, None
    lang = node_text_parts[1]
    if lang not in CONCEPTNET_LANGS:
        return None, lang
    if len(node_text_parts) == 3:
        return node_text_parts[2], lang
    if len(node_text_parts) == 4:
        # consider returning also POS
        return node_text_parts[2], lang
    # logger.error(f"node {node_text} has unexpected strucure - misconception about data structure")
    return None, lang


def parse_conceptnet_relation(relation_text: str):
    relation_text_parts = relation_text.split("/")[1:]
    if relation_text_parts[0] != 'r':
        # logger.error(f"relation {relation_text} starts with other than r prefix - misconception about data structure")
        return None
    if len(relation_text_parts) == 2:  # and relation_text_parts[1] in CONCEPTNET_RELATIONS:
        return relation_text_parts[1]
    if relation_text_parts[1] == 'dbpedia' and len(relation_text_parts) > 2:
        return '_'.join(relation_text_parts[1:])
    # logger.error(f"relation {relation_text} has unexpected strucure - misconception about data structure")
    return None


if __name__ == '__main__':
    conceptnet_path = Path('conceptnet-5.7.0-assertions.csv')
    df = pd.DataFrame(
        list(parse_conceptnet_dump(conceptnet_path)),
        columns=['relation', 'source', 'target', 'lang']
    )
    df.to_csv(conceptnet_path.with_suffix('.en-pl.csv'))
