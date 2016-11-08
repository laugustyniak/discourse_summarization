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


class AspectsGraphBuilder():
    def __addNodeToGraph(self, graph, node):
        if graph.has_node(node):
            graph.node[node]['support'] += 1
        else:
            graph.add_node(node, {'support': 1})

        return graph

    def __addEdgeToGraph(self, graph, nodeLeft, nodeRight):
        if graph.has_edge(nodeLeft, nodeRight):
            graph[nodeLeft][nodeRight]['support'] += 1
        else:
            graph.add_edge(nodeLeft, nodeRight)
            graph[nodeLeft][nodeRight]['support'] = 1

        return graph

    def __buildAspectsGraph(self, rules, aspectsPerEDU):
        graph = nx.DiGraph()

        for rule_id, rule in enumerate(rules):
            # logging.debug('Rule: {}'.format(rule))
            left_node, right_node = rule

            for id1, aspect_left in enumerate(aspectsPerEDU[left_node]):
                for id2, aspect_right in enumerate(aspectsPerEDU[right_node]):
                    graph = self.__addNodeToGraph(graph, aspect_left)
                    graph = self.__addNodeToGraph(graph, aspect_right)

                    graph = self.__addEdgeToGraph(graph, aspect_left, aspect_right)

        return graph

    def __calculateEdgesWeight(self, graph):

        for edge in graph.edges():
            edgeSupport = graph[edge[0]][edge[1]]['support']
            firstNodeSupport = graph.node[edge[0]]['support']
            graph[edge[0]][edge[1]]['weight'] = edgeSupport / float(firstNodeSupport)

            del graph[edge[0]][edge[1]]['support']

        return graph

    def __deleteTemporaryInfo(self, graph):

        for node in graph.nodes():
            del graph.node[node]['support']

        return graph

    def __calculatePageRanks(self, graph):

        pageRanks = nx.pagerank(graph)
        pageRanks = OrderedDict(sorted(pageRanks.items(), key=itemgetter(1), reverse=True))

        return pageRanks

    def build(self, rules, aspectsPerEDU):

        graph = self.__buildAspectsGraph(rules, aspectsPerEDU)
        graph = self.__calculateEdgesWeight(graph)
        graph = self.__deleteTemporaryInfo(graph)

        pageRanks = self.__calculatePageRanks(graph)

        return graph, pageRanks
