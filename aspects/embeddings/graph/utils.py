from pathlib import Path
from typing import Dict, Any, Union

import networkx as nx
import numpy as np
from sklearn import metrics as sk_metrics

from aspects.data_io import serializer
from aspects.embeddings.graph.metrics import map_score, distortion


def nodes_to_mtx(nodes: Dict[Any, Dict], mapping: Dict) -> np.array:
    """Converts nodes shortest paths to matrix form."""
    n_nodes = len(mapping)
    mtx = np.zeros((n_nodes, n_nodes), dtype=np.int)

    for node_1 in nodes.keys():
        for node_2, value in nodes[node_1].items():
            mtx[mapping[node_1]][mapping[node_2]] = value

    return mtx


def preprocess_data(graph: nx.Graph, shortest_paths_path: Union[str, Path]):
    """Reads graph and shortest paths."""
    shortest_paths = serializer.load(shortest_paths_path)
    mapping = dict(zip(
        list(graph.nodes()),
        np.arange(0, graph.number_of_nodes())
    ))

    if nx.is_directed(graph):
        graph = nx.DiGraph(graph)
    else:
        graph = nx.Graph(graph)

    graph = nx.relabel_nodes(graph, mapping)
    graph_dists = nodes_to_mtx(shortest_paths, mapping)
    return graph, graph_dists, mapping


def calculate_reconstruction_metrics(
        graph: Union[nx.DiGraph, nx.Graph],
        graph_dists: np.ndarray,
        embeddings: np.ndarray
):
    emb_dists = sk_metrics.pairwise_distances(
        embeddings, metric='euclidean'
    )

    # Normalize distances on graph and embedding to same scale
    graph_dists = graph_dists / np.max(graph_dists)
    emb_dists = emb_dists / np.max(emb_dists)

    return {
        'mAP': map_score(graph=graph, emb_dists=emb_dists),
        'distortion': distortion(graph_dists=graph_dists, emb_dists=emb_dists)
    }
