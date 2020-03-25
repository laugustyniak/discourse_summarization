import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from aspects.embeddings.metrics import map_score, distortion


def convert_node_dict_to_mtx(input_dict, mapping):
    """Converts nodes shortest paths to matrix form."""
    nb_nodes = len(mapping)
    mtx = np.zeros((nb_nodes, nb_nodes), dtype=np.int)
    for node_1 in input_dict.keys():
        for node_2, value in input_dict[node_1].items():
            mtx[mapping[node_1]][mapping[node_2]] = value
    return mtx


def convert_embedding_to_mtx(emb, mapping):
    """Converts embedding to matrix form."""
    nb_nodes = len(mapping)
    emb_dim = emb.emb_dim

    mtx = np.zeros((nb_nodes, emb_dim), dtype=np.float32)
    e = emb.to_dict()

    for node, val in e.items():
        mtx[mapping[int(node)]] = val
    return mtx


def preprocess_data(graph_path, sp_path):
    """Reads graph and shortest paths."""
    graph = nx.read_gpickle(graph_path)
    sp = pd.read_pickle(sp_path)
    mapping = dict(zip(
        list(graph.nodes()),
        np.arange(0, graph.number_of_nodes())
    ))

    if nx.is_directed(graph):
        graph = nx.DiGraph(graph)
    else:
        graph = nx.Graph(graph)

    graph = nx.relabel_nodes(graph, mapping)
    graph_dists = convert_node_dict_to_mtx(sp, mapping)
    return graph, graph_dists, mapping


def calculate_gr_metrics(graph, graph_dists, emb_mtx):
    """Calculates graph reconstruction metrics."""
    emb_dists = sk_metrics.pairwise_distances(
        emb_mtx, metric='euclidean'
    )

    # Normalize distances on graph and embedding to same scale
    graph_dists = graph_dists / np.max(graph_dists)
    emb_dists = emb_dists / np.max(emb_dists)

    return {
        'mAP': map_score(graph=graph, emb_dists=emb_dists),
        'distortion': distortion(
            graph_dists=graph_dists, emb_dists=emb_dists
        )
    }
