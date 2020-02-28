import logging
from collections import defaultdict, OrderedDict
from typing import Tuple, Dict, Union

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

ASPECT_IMPORTANCE = 'importance'


def extend_graph_nodes_with_sentiments_and_weights(
        graph: nx.MultiDiGraph,
        discourse_trees_df: pd.DataFrame
) -> Tuple[nx.MultiDiGraph, Dict]:
    n_aspects_not_in_graph = 0
    n_aspects_updated = 0
    aspect_sentiments = defaultdict(list)

    for _, row in tqdm(discourse_trees_df.iterrows(), total=len(discourse_trees_df), desc='Aspects and sentiment'):
        for aspects, sentiment in zip(row.aspects, row.sentiment):
            for aspect in aspects:
                aspect_sentiments[aspect].append(sentiment)

    for aspect, sentiments in tqdm(aspect_sentiments.items(), desc='Adding attributes to the graph nodes'):
        try:
            graph.node[aspect]['count'] = len(sentiments)
            graph.node[aspect]['sentiment_avg'] = float(np.average(sentiments))
            graph.node[aspect]['sentiment_sum'] = float(np.sum(sentiments))
            graph.node[aspect][ASPECT_IMPORTANCE] = float(np.sum([x ** 2 for x in sentiments]))
            n_aspects_updated += 1
        except KeyError as err:
            n_aspects_not_in_graph += 1
            log.info('There is not aspect: {} in graph'.format(str(err)))

    log.info('#{} aspects not in graph'.format(n_aspects_not_in_graph))
    log.info('#{} aspects updated in graph'.format(n_aspects_updated))

    return graph, aspect_sentiments


def calculate_moi_by_gerani(
        graph: nx.MultiDiGraph,
        weighted_page_rank: Union[Dict, OrderedDict],
        alpha_coefficient=0.5
) -> nx.MultiDiGraph:
    aspect_importance = nx.get_node_attributes(graph, ASPECT_IMPORTANCE)

    for aspect, weighted_page_rank_element in tqdm(weighted_page_rank.items()):
        if aspect in aspect_importance:
            dir_moi = aspect_importance[aspect]
        else:
            dir_moi = 0

        graph.node[aspect]['moi'] = alpha_coefficient * dir_moi + (1 - alpha_coefficient) * weighted_page_rank_element
        graph.node[aspect]['pagerank'] = weighted_page_rank_element

    return graph
