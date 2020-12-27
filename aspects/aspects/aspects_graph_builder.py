import logging
from collections import OrderedDict
from itertools import product
from operator import itemgetter
from typing import Callable, Union, Dict

import mlflow
import networkx as nx
import pandas as pd
from tqdm import tqdm

from aspects.utilities.settings import setup_mlflow

logger = logging.getLogger(__name__)
setup_mlflow()


class Aspect2AspectGraph:
    def __init__(self, aspects_to_skip=None):
        self.aspects_to_skip = aspects_to_skip or []

    def build(
        self, discourse_tree_df: pd.DataFrame, filter_relation_fn: Callable = None
    ):
        """
        Build aspect(EDU)-aspect(EDU) network based on RST and ConceptNet relation.

        Parameters
        ----------
        discourse_tree_df: pd.DataFrame

        filter_relation_fn:

        Returns
        -------
        graph: networkx.DiGraph
            Graph with aspect-aspect relations

        """
        log_rules_stats(discourse_tree_df)
        if filter_relation_fn:
            tqdm.pandas(desc="Rules filtering...")
            discourse_tree_df.rules = discourse_tree_df.rules.progress_apply(
                filter_relation_fn
            )
            log_rules_stats(discourse_tree_df, "_filtered")

        return self.build_aspects_graph(discourse_tree_df)

    def build_aspects_graph(self, discourse_tree_df: pd.DataFrame) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for _, row in tqdm(
            discourse_tree_df.iterrows(),
            total=len(discourse_tree_df),
            desc="Generating aspect-aspect graph based on rules",
        ):
            for edu_left, edu_right, relation, weight in row.rules:
                for aspect_left, aspect_right in product(
                    row.aspects[edu_left], row.aspects[edu_right]
                ):
                    if not (
                        aspect_left in self.aspects_to_skip
                        or aspect_right in self.aspects_to_skip
                    ):
                        graph = self.add_aspects_to_graph(
                            graph, aspect_left, aspect_right, relation, weight
                        )
        return graph

    @staticmethod
    def add_aspects_to_graph(graph, aspect_left, aspect_right, relation, weight):
        if aspect_left != aspect_right:
            logger.debug(f"Add rule: {(aspect_left, aspect_right, relation, weight)}")
            graph.add_edge(
                aspect_left, aspect_right, relation_type=relation, weight=weight
            )
        return graph


def log_rules_stats(discourse_tree_df, suffix: str = ""):
    rules_cardinality = discourse_tree_df.rules.apply(lambda rs: len(rs))
    mlflow.log_params(
        {
            f"rules_per_tree_avg{suffix}": rules_cardinality.mean(),
            f"rules_per_tree_min{suffix}": rules_cardinality.min(),
            f"rules_per_tree_max{suffix}": rules_cardinality.max(),
            f"rules_per_tree_median{suffix}": rules_cardinality.median(),
            f"rules_per_tree_std{suffix}": rules_cardinality.std(),
            f"rules_per_tree_sum{suffix}": rules_cardinality.sum(),
        }
    )


def merge_multiedges(
    graph: object,
    node_attrib_name: object = "weight",
    default_node_weight: float = 1,
    aspect_cluster_map: Dict[str, str] = None,
) -> nx.Graph:
    """
    Merge multiple edges between nodes into one relation and sum attribute weight.

    Merge the edges connecting two nodes and consider the sum of their weights as the weight of the merged graph.
    We also ignore the relation direction for the purpose of generating the tree.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Aspect-aspect relation graph.

    node_attrib_name : str
        Name of node's attribute that will be summed up in merged relations.

    Returns
    -------
    graph_new : networkx.Graph()
        Aspect-aspect graph with maximum single relation between each pair of nodes.

    Args:
        default_node_weight:

    """
    logger.info("Create a new graph without multiple edges between nodes.")
    graph_new = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data[node_attrib_name] if node_attrib_name in data else default_node_weight
        if graph_new.has_edge(u, v):
            graph_new[u][v][node_attrib_name] += w
        else:
            graph_new.add_edge(u, v, weight=w)

    logger.info("Copy nodes attributes from multi edge graph to flattened one.")
    nx.set_node_attributes(graph_new, dict(graph.nodes.items()))

    return graph_new


def calculate_weighted_page_rank(
    graph: Union[nx.MultiDiGraph, nx.MultiGraph, nx.Graph, nx.DiGraph],
    weight: str = "weight",
) -> OrderedDict:
    """
    Calculate Page Rank for ARRG.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph of aspect-aspect relation ARRG.

    weight : str, optional
        Name of edge attribute that consists of weight for an edge. it is
        used to calculate Weighted version of Page Rank.

    Returns
    -------
    page_ranks : OrderedDict
        Page Rank values for ARRG.

    """
    logger.info("Weighted Page Rank calculation starts.")
    page_ranks = nx.pagerank_scipy(graph, weight=weight)
    logger.info("Weighted Page Rank calculation ended.")
    return OrderedDict(sorted(page_ranks.items(), key=itemgetter(1), reverse=True))


def calculate_hits(
    graph: Union[nx.MultiDiGraph, nx.MultiGraph, nx.Graph, nx.DiGraph]
) -> OrderedDict:
    logger.info("HITS calculation starts.")
    hubs, authorities = nx.hits_scipy(graph)
    logger.info("HITS calculation ended.")
    return OrderedDict(sorted(authorities.items(), key=itemgetter(1), reverse=True))


def sort_networkx_attributes(graph_attribs_tuples):
    return sorted(
        list(graph_attribs_tuples),
        key=lambda node_attrib_pair: node_attrib_pair[1],
        reverse=True,
    )
