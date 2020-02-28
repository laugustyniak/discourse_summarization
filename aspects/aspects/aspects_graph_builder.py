import logging
from collections import OrderedDict
from itertools import product
from operator import itemgetter
from typing import Callable, Union

import networkx as nx
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
log = logging.getLogger(__name__)


class Aspect2AspectGraph:
    def __init__(self, with_cycles_between_aspects=False):
        self.with_cycles_between_aspects = with_cycles_between_aspects

    def build(self, discourse_tree_df: pd.DataFrame, conceptnet_io: bool = False, filter_relation_fn: Callable = None):
        """
        Build aspect(EDU)-aspect(EDU) network based on RST and ConceptNet relation.

        Parameters
        ----------
        discourse_tree_df: pd.DataFrame

        filter_relation_fn:

        conceptnet_io: bool
            Do we use ConceptNet.io relation in graph?

        Returns
        -------
        graph: networkx.DiGraph
            Graph with aspect-aspect relations

        """
        if filter_relation_fn:
            discourse_tree_df.rules = discourse_tree_df.rules.progress_apply(filter_relation_fn)
        return self.build_aspects_graph(discourse_tree_df)

    def build_aspects_graph(self, discourse_tree_df: pd.DataFrame) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for row_id, row in tqdm(
                discourse_tree_df.iterrows(),
                total=len(discourse_tree_df),
                desc='Generating aspect-aspect graph based on rules'
        ):
            for edu_left, edu_right, relation, gerani_weight in row.rules:
                for aspect_left, aspect_right in product(row.aspects[edu_left], row.aspects[edu_right]):
                    graph = self.add_aspects_to_graph(graph, aspect_left, aspect_right, relation, gerani_weight)
        return graph

    def add_aspects_to_graph(self, graph, aspect_left, aspect_right, relation, gerani_weight):
        if self.with_cycles_between_aspects or aspect_left != aspect_right:
            log.info(f'Add rule: {(aspect_left, aspect_right, relation, gerani_weight)}')
            graph.add_edge(aspect_left, aspect_right, relation_type=relation, gerani_weight=gerani_weight)
        return graph

    # def filter_gerani(self, rules):
    #     """
    #     Filter rules by its confidence,
    #
    #     Parameters
    #     ----------
    #     rules : dict
    #         Dictionary of document id and list of rules. Relation tuple
    #         Relation(aspect_right, aspect, relation, gerani_weight)
    #
    #     Returns
    #     -------
    #     rules_filtered : defaultdict
    #         Dictionary of document id (-1 as indicator for all documents)
    #         and list of rules/relations between aspects with their maximum
    #         gerani weights.
    #
    #         Examples
    #         {-1: [Relation(aspect1=u'screen', aspect2=u'phone',
    #                                     relation_type='Elaboration',
    #                                     gerani_weight=1.38),
    #                            Relation(aspect1=u'speaker', aspect2=u'sound',
    #                                     relation_type='Elaboration',
    #                                     gerani_weight=0.29)]}
    #     """
    #     rule_per_doc = defaultdict(list)
    #     for doc_id, rules_list in rules.iteritems():
    #         rules_filtered = []
    #         for rule in rules_list:
    #             log.debug('Rule: {}'.format(rule))
    #             left_node, right_node, relation, gerani_weight = rule
    #             for aspect_left, aspect_right in self.aspects_iterator(left_node, right_node):
    #                 rules_filtered.append(AspectsRelation(aspect_left, aspect_right, relation, gerani_weight))
    #         if len(rules_filtered):
    #             relation_counter = Counter([x[:3] for x in rules_filtered])
    #             rule_confidence = sorted(relation_counter, key=operator.itemgetter(1), reverse=True)[0]
    #             rules_confidence = [r for r in rules_filtered if rule_confidence == r[:3]]
    #             rule_per_doc[doc_id].extend(
    #                 [
    #                     max(v, key=lambda rel: rel.gerani_weight)
    #                     for g, v
    #                     in groupby(sorted(rules_confidence), key=lambda rel: rel[:3])
    #                 ]
    #             )
    #         else:
    #             log.info('Empty rule list for document {}'.format(doc_id))
    #     relations_list = [
    #         (
    #                 group + (sum([rel.gerani_weight for rel in relations]),)
    #         )
    #         for group, relations
    #         in groupby(sorted(flatten(rule_per_doc.values())), key=lambda rel: rel[:3])
    #     ]
    #     # map relations into namedtuples
    #     relations_list = [
    #         AspectsRelation(a1, a2, r, w)
    #         for a1, a2, r, w
    #         in relations_list
    #     ]
    #     rules = {-1: relations_list}
    #     return rules


def merge_multiedges(graph: object, node_attrib_name: object = 'weight') -> object:
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

    """
    graph_new = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data[node_attrib_name] if node_attrib_name in data else 1.0
        if graph_new.has_edge(u, v):
            graph_new[u][v][node_attrib_name] += w
        else:
            graph_new.add_edge(u, v, weight=w)
    return graph_new


def extract_spanning_tree(graph: Union[nx.Graph], weight: str = 'weight', max_number_of_nodes: int = None) -> nx.Graph:
    """
    To obtain a hierarchical tree structure from the extracted subgraph (filtered by moi measure). We find
    the Maximum Spanning Tree of the undirected subgraph and set the highest weighted aspect as the root of the
    tree. This process results in a useful knowledge structure of aspects with their associated weight and sentiment
    polarity connected with the rhetorical relations called Aspect Hierarchical Tree (AHT).

    Parameters
    ----------
    graph : nx.Graph
        Graph with merged relations between aspects to sum of their weights - merge_multiedges_in_arrg method.
    weight : str
        Name of weight attribute for maximum spanning tree.
    max_number_of_nodes : int

    """
    spanning_tree = nx.maximum_spanning_tree(graph, weight=weight)
    sorted_nodes = sorted(spanning_tree.degree, key=lambda node_degree_pair: node_degree_pair[1], reverse=True)
    return spanning_tree.subgraph(sorted_nodes[:max_number_of_nodes].nodes)


def calculate_weighted_page_rank(
        graph: Union[nx.MultiDiGraph, nx.MultiGraph, nx.Graph, nx.DiGraph],
        weight='weight'
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
    log.info('Weighted Page Rank calculation starts here.')
    page_ranks = nx.pagerank_scipy(graph, weight=weight)
    log.info('Weighted Page Rank calculation ended.')
    return OrderedDict(sorted(page_ranks.items(), key=itemgetter(1), reverse=True))
