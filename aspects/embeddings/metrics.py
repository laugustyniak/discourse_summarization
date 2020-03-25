"""Metrics for graph reconstruction."""

import networkx as nx
import numpy as np


def map_score(graph: nx.Graph, emb_dists: np.ndarray) -> float:
    """Calculates mean average precision on embeddings.

    Borrowed from https://arxiv.org/pdf/1705.08039.pdf
    Reference repo: https://github.com/facebookresearch/poincare-embeddings/
    :param graph: Input Graph
    :param emb_dists: Input distances matrix between nodes on embedding
    :return: MAP (Mean Average Precision)
    """

    def resolve_indices(node_id):
        """Resolve indices after removal of node_id in array."""
        mapping = {}
        for j in range(nb_nodes):
            if j < node_id:
                mapping[j] = j
            else:
                mapping[j + 1] = j
        return mapping

    nb_nodes = graph.number_of_nodes()
    _map = 0.0
    _nodes_with_neighbors = 0

    for i in range(nb_nodes):
        neighbors = list(graph.neighbors(i))

        if i in neighbors:
            neighbors.remove(i)

        if len(neighbors) > 0:
            dists = np.delete(emb_dists[i], i)
            ranks = np.argsort(np.argsort(dists)) + 1

            old_indices = resolve_indices(i)

            neighbors_ranks = np.sort([
                ranks[old_indices[nid]] for nid in neighbors
            ])
            precision = [
                (j + 1) / rank
                for j, rank in enumerate(neighbors_ranks)
            ]

            _map += np.average(precision)
            _nodes_with_neighbors += 1

    return _map / _nodes_with_neighbors


def distortion(graph_dists: np.ndarray, emb_dists: np.ndarray) -> float:
    """Calculates distortion metric between embedding and original graph.

    Borrowed from https://arxiv.org/pdf/1804.03329.pdf
    Reference repo: https://github.com/HazyResearch/hyperbolics/
    :param graph_dists: Input distances matrix between nodes on graph
    :param emb_dists: Input distances matrix between nodes on embedding
    :return: Distortion metric
    """
    assert len(graph_dists) == len(emb_dists), "Distances matrices " \
                                               "have to be equal length"
    nodes_count = len(graph_dists)
    d = 0.0
    for node in range(nodes_count):
        ndg = graph_dists[node]
        nde = emb_dists[node]

        # Ignore distances between same nodes
        ndg[node] = 1.0
        nde[node] = 1.0

        # Ignore not connected nodes
        if 0 in ndg:
            for idx, it in enumerate(ndg):
                if it == 0:
                    nde[idx] = 1.0
                    ndg[idx] = 1.0
                    nodes_count -= 1

        d += np.sum((np.abs(nde - ndg)) / ndg)
    return d / (nodes_count * (nodes_count - 1))
