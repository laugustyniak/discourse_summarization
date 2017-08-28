# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda
import logging
import sys

import networkx as nx
from collections import OrderedDict
from operator import itemgetter

log = logging.getLogger(__name__)


class AspectsGraphBuilder(object):
    def __init__(self):
        pass

    def _add_node_to_graph(self, graph, node):
        if graph.has_node(node):
            graph.node[node]['support'] += 1
        else:
            graph.add_node(node, {'support': 1})

        return graph

    def _add_edge_to_graph(self, graph, node_left, node_right):
        if graph.has_edge(node_left, node_right):
            graph[node_left][node_right]['support'] += 1
        else:
            graph.add_edge(node_left, node_right)
            graph[node_left][node_right]['support'] = 1

        return graph

    def _build_aspects_graph(self, rules, aspects_per_edu):
        graph = nx.DiGraph()

        for rule_id, rule in enumerate(rules):
            log.debug('Rule: {}'.format(rule))
            left_node, right_node, relations = rule

            for id1, aspect_left in enumerate(aspects_per_edu[left_node]):
                for id2, aspect_right in enumerate(aspects_per_edu[right_node]):
                    graph = self._add_node_to_graph(graph, aspect_left)
                    graph = self._add_node_to_graph(graph, aspect_right)

                    graph = self._add_edge_to_graph(graph, aspect_left,
                                                    aspect_right)

        return graph

    def _calculate_edges_weight(self, graph):

        for edge in graph.edges():
            edge_support = graph[edge[0]][edge[1]]['support']
            first_node_support = graph.node[edge[0]]['support']
            graph[edge[0]][edge[1]]['weight'] = \
                edge_support / float(first_node_support)

            del graph[edge[0]][edge[1]]['support']

        return graph

    def _delete_temporary_info(self, graph):

        for node in graph.nodes():
            del graph.node[node]['support']

        return graph

    def _calculate_page_ranks(self, graph):

        page_ranks = nx.pagerank(graph)
        page_ranks = OrderedDict(sorted(page_ranks.items(), key=itemgetter(1),
                                        reverse=True))

        return page_ranks

    def build(self, rules, aspects_per_edu):

        graph = self._build_aspects_graph(rules, aspects_per_edu)
        graph = self._calculate_edges_weight(graph)
        graph = self._delete_temporary_info(graph)

        page_ranks = self._calculate_page_ranks(graph)

        return graph, page_ranks
