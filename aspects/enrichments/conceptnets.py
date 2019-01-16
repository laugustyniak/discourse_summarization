import logging
import pickle

import pandas as pd

try:
    from repoze.lru import lru_cache
except:
    from functools import lru_cache

from aspects.data.sentic.senticnet5 import senticnet
from aspects.utilities import settings

log = logging.getLogger(__name__)

SEMANTIC_COL_NAME = ['semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5']


@lru_cache(None)
def load_sentic():
    sentic_df = pd.DataFrame.from_dict(senticnet, orient='index')
    sentic_df.columns = ['pleasantness', 'attention', 'sensivity', 'aptitude', 'modtag1', 'modtag2',
                         'polarity_value', 'polarity_intensity',
                         'semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5']
    return sentic_df


def get_concept_from_senticnet_by_partname(sentic_df, partname):
    """
    Get part of data frame with all concept's data related to partname
    concept partname could be regex
    :param sentic_df: pd.DataFrame
        Data frame with sentic's concepts
    :param partname: concept that will be filtered
    :return: Data Frame with concepts

    Parameters
    ----------
    sentic_df
    """
    return sentic_df[sentic_df.index.str.contains(partname)].sort_index()


def get_semantic_concept_by_concept(sentic_df, partname, exact_match=False):
    """
    Get concept and list of related concepts related to partname concept,
    partname could be regex

    :param sentic_df: pd.DataFrame
        Data frame with sentic's concepts
    :param exact_match: bool
        Do we want to find exactly same concepts? Otherwise we will get
        all concepts with even substring of partname concept.
    :param partname: str
        Concept that will be filtered.

    :return: dict
        Concepts dictionary key: concept name,
        values: list of related concepts
    """
    concepts = {}
    if exact_match:
        df = sentic_df[sentic_df.index == partname].sort_index()
    else:
        df = sentic_df[sentic_df.index.str.contains(partname)].sort_index()
    for row in df.iterrows():
        concepts[row[0]] = [row[1][col] for col in SEMANTIC_COL_NAME]
    return concepts


@lru_cache(maxsize=None)
def load_conceptnet_io():
    log.info('ConceptNet.io temp files will be load from: {}'.format(settings.CONCEPTNET_IO_PKL))
    with open(settings.CONCEPTNET_IO_PKL.as_posix(), 'rb') as f:
        conceptnet_io = pickle.load(f)
    return conceptnet_io


def get_concept_neighbours_by_relation_type(
        conceptnet,
        concept,
        relation_types_get_child,
        relation_types_get_parent,
        neighbour_relations,
        level=1
):
    neighbours = get_neighbours_child_and_parents(
        conceptnet, concept, relation_types_get_child, relation_types_get_parent)
    # print('First level neighbours for {}: {}'.format(concept, len(neighbours)))

    neighbours_level = []
    for l in range(1, level):
        for neighbour in neighbours:
            neighbours_level += get_neighbours_child_and_parents(
                conceptnet,
                neighbour,
                relation_types_get_child + neighbour_relations,
                relation_types_get_parent + neighbour_relations
            )
        neighbours = set(neighbours_level)
        # print('{} level neighbours: {}'.format(level, len(neighbours)))
    return set(neighbours)


def get_neighbours_child_and_parents(conceptnet, concept, relation_types_get_child, relation_types_get_parent):
    neighbours_childs = set(
        concept_info['end']
        for concept_info
        in conceptnet[concept]
        if concept_info['relation'] in relation_types_get_child
    )
    neighbours_parents = set(
        concept_info['start']
        for concept_info
        in conceptnet[concept]
        if concept_info['relation'] in relation_types_get_parent
    )

    return list(neighbours_childs.union(neighbours_parents))
