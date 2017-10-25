import logging

from collections import defaultdict

import networkx as nx

log = logging.getLogger(__name__)


def get_dir_moi_for_node(graph, aspects_per_edu, documents_info):
    n_skipped_edus = 0
    n_aspects_not_in_graph = 0
    n_aspects_updated = 0
    n_all_documents = len(documents_info)
    aspect_sentiments = defaultdict(list)

    if not isinstance(aspects_per_edu, dict):
        aspects_per_edu = dict(aspects_per_edu)
    for doc_info in documents_info.itervalues():
        for edu_id, sentiment in doc_info['sentiment'].iteritems():
            try:
                for aspect in aspects_per_edu[edu_id]:
                    aspect_sentiments[aspect].append(sentiment)
            except KeyError as err:
                n_skipped_edus += 1
                log.info(
                    'Aspect: {} not extracted from edu: {}'.format(str(err),
                                                                   edu_id))
    for aspect, sentiments in aspect_sentiments.iteritems():
        try:
            # fixme GEXF write error with list
            # graph.node[aspect]['sentiment'] = sentiments
            graph.node[aspect]['dir_moi'] = int(
                sum([x ** 2 for x in sentiments]))
            n_aspects_updated += 1
        except KeyError as err:
            n_aspects_not_in_graph += 1
            log.info('There is not aspect: {} in graph'.format(str(err)))

    log.info('#{} skipped aspects out of #{} documents'.format(n_skipped_edus,
                                                               n_all_documents))
    log.info('#{} aspects not in graph'.format(n_aspects_not_in_graph))
    log.info('#{} aspects updated in graph'.format(n_aspects_updated))

    return graph


def calculate_moi_by_gerani(graph, alpha=0.5):
    """

    Parameters
    ----------
    graph : nx.DiGraph
        Graph of aspect-aspect relations with weights.

    alpha : float
        Alpha parameter for moi function. 0.5 as default.

    Returns
    -------
    graph : nx.DiGraph
        Graph of aspect-aspect relations with weights extended with moi
        attribute for each node.

    aspect_moi : defaultdict
        Dictionary with aspects as keys and moi weight as values.
    """
    dir_moi = 0
    wprs = nx.pagerank(graph, weight='gerani_weight')
    log.info('Weighted Page Rank, Gerani weight calculated')
    aspect_dir_moi = nx.get_node_attributes(graph, 'dir_moi')
    aspect_moi = defaultdict(float)
    for aspect, wpr in wprs.iteritems():
        if aspect in aspect_dir_moi:
            dir_moi = aspect_dir_moi[aspect]
        moi = alpha * dir_moi + (1 - alpha) * wpr
        aspect_moi[aspect] = moi
        graph.node[aspect]['moi'] = moi
    return graph, aspect_moi
