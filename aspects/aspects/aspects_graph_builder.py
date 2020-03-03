import logging
from collections import OrderedDict
from itertools import product
from operator import itemgetter
from typing import Callable, Union

import networkx as nx
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
loger = logging.getLogger(__name__)


class Aspect2AspectGraph:

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
        if aspect_left != aspect_right:
            loger.info(f'Add rule: {(aspect_left, aspect_right, relation, gerani_weight)}')
            graph.add_edge(aspect_left, aspect_right, relation_type=relation, gerani_weight=gerani_weight)
        return graph


def merge_multiedges(graph: object, node_attrib_name: object = 'weight', default_node_weight: float = 1) -> nx.Graph:
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
    loger.info('Create a new graph without multiple edges between nodes.')
    graph_new = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data[node_attrib_name] if node_attrib_name in data else default_node_weight
        if graph_new.has_edge(u, v):
            graph_new[u][v][node_attrib_name] += w
        else:
            graph_new.add_edge(u, v, weight=w)

    loger.info('Copy nodes attributes from multi edge graph to flattened one.')
    nx.set_node_attributes(graph_new, dict(graph.nodes.items()))

    return graph_new


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
    loger.info('Weighted Page Rank calculation starts.')
    page_ranks = nx.pagerank_scipy(graph, weight=weight)
    loger.info('Weighted Page Rank calculation ended.')
    return OrderedDict(sorted(page_ranks.items(), key=itemgetter(1), reverse=True))


# def filter_rules_gerani(rules_rows: List[EDURelation]) -> List[EDURelation]:
#     """
#     Filter rules by its confidence,
#
#     Parameters
#     ----------
#     rules_rows : dict
#         Dictionary of document id and list of rules. Relation tuple
#         Relation(aspect_right, aspect, relation, gerani_weight)
#
#     Returns
#     -------
#     rules_filtered : list
#         List of rules/relations between aspects with their maximum gerani weights.
#
#         Examples
#         [
#             Relation(aspect1=u'screen', aspect2=u'phone', relation_type='Elaboration', gerani_weight=1.38),
#             Relation(aspect1=u'speaker', aspect2=u'sound', relation_type='Elaboration', gerani_weight=0.29)
#         ]
#     """
#     for rules in rules_rows:
#         rules_filtered = []
#         for rule in rules:
#             left_node, right_node, relation, weight = rule
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
#             logging.info('Empty rule list for document {}'.format(doc_id))
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


def sort_networkx_attibutes(graph_attribs_tuples):
    return sorted(list(graph_attribs_tuples), key=lambda node_attrib_pair: node_attrib_pair[1], reverse=True)
