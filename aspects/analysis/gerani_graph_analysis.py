import logging
from collections import defaultdict, OrderedDict
from typing import Tuple, Dict, Union

import mlflow
import networkx as nx
import numpy as np
import pandas as pd
from toolz import pluck
from tqdm import tqdm

from aspects.aspects.aspects_graph_builder import (
    calculate_weighted_page_rank,
    merge_multiedges,
    calculate_hits,
)
from aspects.utilities.settings import setup_mlflow

setup_mlflow()

logger = logging.getLogger(__name__)

ASPECT_IMPORTANCE = "importance"


def extend_graph_nodes_with_sentiments_and_weights(
    graph: nx.MultiDiGraph, discourse_trees_df: pd.DataFrame
) -> Tuple[nx.MultiDiGraph, Dict]:
    n_aspects_not_in_graph = 0
    n_aspects_updated = 0
    aspect_sentiments = defaultdict(list)

    for _, row in tqdm(
        discourse_trees_df.iterrows(),
        total=len(discourse_trees_df),
        desc="Adding aspects and sentiment to the graph",
    ):
        for aspects, sentiment in zip(row.aspects, row.sentiment):
            for aspect in aspects:
                aspect_sentiments[aspect].append(sentiment)

    for aspect, sentiments in tqdm(
        aspect_sentiments.items(), desc="Adding attributes to the graph nodes"
    ):
        try:
            graph.nodes[aspect]["count"] = len(sentiments)
            graph.nodes[aspect]["sentiment_avg"] = float(np.average(sentiments))
            graph.nodes[aspect]["sentiment_sum"] = float(np.sum(sentiments))
            graph.nodes[aspect][ASPECT_IMPORTANCE] = float(
                np.sum([x ** 2 for x in sentiments])
            )
            n_aspects_updated += 1
        except KeyError as err:
            n_aspects_not_in_graph += 1
            logger.info("There is not aspect: {} in graph".format(str(err)))

    logger.info("#{} aspects not in graph".format(n_aspects_not_in_graph))
    logger.info("#{} aspects updated in graph".format(n_aspects_updated))

    return graph, aspect_sentiments


def calculate_moi_by_gerani(
    graph: nx.MultiDiGraph,
    weighted_page_rank: Union[Dict, OrderedDict],
    alpha_coefficient=0.5,
) -> nx.MultiDiGraph:
    aspect_importance = nx.get_node_attributes(graph, ASPECT_IMPORTANCE)

    for aspect, weighted_page_rank_element in tqdm(
        weighted_page_rank.items(), desc="Calculating moi by Gerani..."
    ):
        if aspect in aspect_importance:
            dir_moi = aspect_importance[aspect]
        else:
            dir_moi = 0

        graph.nodes[aspect]["moi"] = (
            alpha_coefficient * dir_moi
            + (1 - alpha_coefficient) * weighted_page_rank_element
        )
        graph.nodes[aspect]["pagerank"] = weighted_page_rank_element

    return graph


def gerani_paper_arrg_to_aht(
    graph: nx.MultiDiGraph,
    max_number_of_nodes: int = 100,
    weight: str = "moi",
    alpha_coefficient: float = 0.5,
) -> nx.Graph:
    logger.info("Generate Aspect Hierarchical Tree based on ARRG")
    aspects_weighted_page_rank = calculate_weighted_page_rank(graph, "weight")
    graph = calculate_moi_by_gerani(
        graph=graph,
        weighted_page_rank=aspects_weighted_page_rank,
        alpha_coefficient=alpha_coefficient,
    )

    graph_flatten = merge_multiedges(graph)
    sorted_nodes = sorted(
        list(aspects_weighted_page_rank.items()),
        key=lambda node_degree_pair: node_degree_pair[1],
        reverse=True,
    )
    csv_name = "/tmp/gerani_page_ranks.csv"
    pd.DataFrame(sorted_nodes, columns=["aspect", "page_rank"]).to_csv(csv_name)
    mlflow.log_artifact(csv_name)
    top_nodes = list(pluck(0, sorted_nodes[:max_number_of_nodes]))
    sub_graph = graph_flatten.subgraph(top_nodes)
    maximum_spanning_tree = nx.maximum_spanning_tree(sub_graph, weight=weight)
    nx.set_node_attributes(maximum_spanning_tree, dict(sub_graph.nodes.items()))
    return maximum_spanning_tree


def our_paper_arrg_to_aht(
    graph: nx.MultiDiGraph,
    max_number_of_nodes: int,
    weight: str = "weight",
    alpha_coefficient: float = 0.5,
    use_aspect_clusters: bool = False,
) -> nx.Graph:
    logger.info("Generate Aspect Hierarchical Tree based on ARRG")
    # aspects_rank = calculate_hits(graph)
    aspects_rank = nx.in_degree_centrality(graph)
    # aspects_rank = calculate_weighted_page_rank(graph, 'weight')
    graph = calculate_moi_by_gerani(
        graph=graph,
        weighted_page_rank=aspects_rank,
        alpha_coefficient=alpha_coefficient,
    )
    if use_aspect_clusters:
        aspect_cluster_map = {}
    else:
        aspect_cluster_map = None
    # TODO add aspect merging here
    graph_flatten = merge_multiedges(graph, aspect_cluster_map=aspect_cluster_map)
    # sorted_nodes = sorted(list(graph_flatten.degree()), key=lambda node_degree_pair: node_degree_pair[1], reverse=True)
    sorted_nodes = sorted(
        list(aspects_rank.items()), key=lambda node_value: node_value[1], reverse=True
    )

    csv_name = "/tmp/our_ranks.csv"
    pd.DataFrame(sorted_nodes, columns=["aspect", "rank"]).to_csv(csv_name)
    mlflow.log_artifact(csv_name)

    top_nodes = list(pluck(0, sorted_nodes[:max_number_of_nodes]))
    sub_graph = graph_flatten.subgraph(top_nodes)
    maximum_spanning_tree = nx.maximum_spanning_tree(sub_graph, weight=weight)
    nx.set_node_attributes(maximum_spanning_tree, dict(sub_graph.nodes.items()))
    return maximum_spanning_tree
