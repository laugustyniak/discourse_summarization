# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import networkx as nx
from collections import OrderedDict
from operator import itemgetter

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


class AspectsGraphBuilder(object):
    def __init__(self):
        pass

    def __add_node_to_graph(self, graph, node):
        if graph.has_node(node):
            graph.node[node]['support'] += 1
        else:
            graph.add_node(node, {'support': 1})

        return graph

    def __add_edge_to_graph(self, graph, node_left, node_right):
        if graph.has_edge(node_left, node_right):
            graph[node_left][node_right]['support'] += 1
        else:
            graph.add_edge(node_left, node_right)
            graph[node_left][node_right]['support'] = 1

        return graph

    def __build_aspects_graph(self, rules, aspects_per_edu):
        graph = nx.DiGraph()

        for rule_id, rule in enumerate(rules):
            # logging.debug('Rule: {}'.format(rule))
            left_node, right_node = rule

            for id1, aspect_left in enumerate(aspects_per_edu[left_node]):
                for id2, aspect_right in enumerate(aspects_per_edu[right_node]):
                    graph = self.__add_node_to_graph(graph, aspect_left)
                    graph = self.__add_node_to_graph(graph, aspect_right)

                    graph = self.__add_edge_to_graph(graph, aspect_left, aspect_right)

        return graph

    def __calculate_edges_weight(self, graph):

        for edge in graph.edges():
            edge_support = graph[edge[0]][edge[1]]['support']
            first_node_support = graph.node[edge[0]]['support']
            graph[edge[0]][edge[1]]['weight'] = edge_support / float(first_node_support)

            del graph[edge[0]][edge[1]]['support']

        return graph

    def __delete_temporary_info(self, graph):

        for node in graph.nodes():
            del graph.node[node]['support']

        return graph

    def __calculate_page_ranks(self, graph):

        page_ranks = nx.pagerank(graph)
        page_ranks = OrderedDict(sorted(page_ranks.items(), key=itemgetter(1), reverse=True))

        return page_ranks

    def build(self, rules, aspects_per_edu):

        graph = self.__build_aspects_graph(rules, aspects_per_edu)
        graph = self.__calculate_edges_weight(graph)
        graph = self.__delete_temporary_info(graph)

        page_ranks = self.__calculate_page_ranks(graph)

        return graph, page_ranks
