import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def extend_graph_nodes_with_sentiments_and_weights(graph, discourse_trees_df: pd.DataFrame):
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
            graph.node[aspect]['dir_moi'] = float(np.sum([x ** 2 for x in sentiments]))
            n_aspects_updated += 1
        except KeyError as err:
            n_aspects_not_in_graph += 1
            log.info('There is not aspect: {} in graph'.format(str(err)))

    log.info('#{} aspects not in graph'.format(n_aspects_not_in_graph))
    log.info('#{} aspects updated in graph'.format(n_aspects_updated))

    return graph


def calculate_moi_by_gerani(graph, alpha=0.5, max_iter=1000):
    """
    Calculate moi metric used by Gerani, S., Mehdad, Y., Carenini, G., Ng, R. T., & Nejat, B. (2014).
    Abstractive Summarization of Product Reviews Using Discourse Structure. Emnlp, 1602-1613.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph of aspect-aspect relations with weights.

    alpha : float
        Alpha parameter for moi function. 0.5 as default.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    Returns
    -------
    graph : nx.DiGraph
        Graph of aspect-aspect relations with weights extended with moi
        attribute for each node.

    aspect_moi : defaultdict
        Dictionary with aspects as keys and moi weight as values.
    """
    dir_moi = 0
    wprs = nx.pagerank_scipy(graph, weight='gerani_weight', max_iter=max_iter)
    log.info('Weighted Page Rank, Gerani weight calculated')
    log.info('Graph with #{} nodes and #{} edges'.format(len(graph.nodes()), len(graph.edges())))
    aspect_dir_moi = nx.get_node_attributes(graph, 'dir_moi')
    aspect_moi = defaultdict(float)
    n_wprs = len(wprs)
    for aspect, wpr in tqdm(wprs.iteritems(), total=n_wprs):
        if aspect in aspect_dir_moi:
            dir_moi = aspect_dir_moi[aspect]
        moi = alpha * dir_moi + (1 - alpha) * wpr
        aspect_moi[aspect] = moi
        graph.node[aspect]['moi'] = moi
        graph.node[aspect]['pagerank'] = wpr
    return graph, aspect_moi
