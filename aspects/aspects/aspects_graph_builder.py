import logging

from collections import OrderedDict
from operator import itemgetter

import networkx as nx

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

    def _add_edge_to_graph(self, graph, node_left, node_right,
                           relation_type='None'):
        if graph.has_edge(node_left, node_right):
            graph[node_left][node_right]['support'] += 1
        else:
            graph.add_edge(node_left, node_right)
            graph[node_left][node_right]['support'] = 1

        graph[node_left][node_right]['relation_type'] = relation_type

        return graph

    def _build_aspects_graph(self, rules, aspects_per_edu):
        """
        Build graph based on list of tuples with apsects ids.

        :param rules: list
            (aspect_id_1, aspect_id_2, relation, weight_dict)

        :param aspects_per_edu: dict
            Dict with id of edu and aspects extracted to this edu.

        :return: networkx.DiGraph()
            Graph with aspect-aspect relation.
        """
        graph = nx.DiGraph()

        for rule_id, rule in enumerate(rules):
            log.debug('Rule: {}'.format(rule))
            left_node, right_node, relation, weigths = rule

            aspects_per_edu = dict(aspects_per_edu)

            # for all aspects from one edu list
            try:
                for aspect_left in aspects_per_edu[left_node]:
                    # and for all aspects from the other edu list
                    for aspect_right in aspects_per_edu[right_node]:
                        graph = self._add_node_to_graph(graph, aspect_left)
                        graph = self._add_node_to_graph(graph, aspect_right)
                        graph = self._add_edge_to_graph(graph, aspect_left,
                                                        aspect_right,
                                                        relation_type=relation)
            except KeyError as err:
                log.info('Lack of aspect: {}'.format(str(err)))
                
        return graph

    def _calculate_edges_weight(self, graph):

        for edge in graph.edges():
            edge_support = graph[edge[0]][edge[1]]['support']
            first_node_support = graph.node[edge[0]]['support']
            # todo: describe method and lines
            # todo: add reference to paper and equations
            graph[edge[0]][edge[1]]['weight'] = \
                edge_support / float(first_node_support)

            del graph[edge[0]][edge[1]]['support']

        return graph

    def _delete_temporary_support_info(self, graph):

        for node in graph.nodes():
            del graph.node[node]['support']

        return graph

    def _calculate_page_ranks(self, graph):

        page_ranks = nx.pagerank(graph)
        page_ranks = OrderedDict(sorted(page_ranks.items(), key=itemgetter(1),
                                        reverse=True))

        return page_ranks

    def build(self, rules, aspects_per_edu, documents_info,
              conceptnet_io=False):
        """
        Build aspect(EDU)-aspect(EDU) network based on RST and ConceptNet
        relation.

        :param rules: tuple
            List of rules that will be used to create aspect-aspect graph,
            list elements: (node_1, node_2, weight).

        :param aspects_per_edu: dict
            Dictionary with edu id and it's aspects.

        :param documents_info: dict
            Dictionary with information about each edu.

        :param conceptnet_io: bool
            Do we use ConcetNet.io relation in graph?

        :return:
            graph: networkx.Graph
                Graph with aspect-aspect relations

            page_rank: networkx.PageRank
                PageRank counted for aspect-aspect graph.
        """

        graph = self._build_aspects_graph(rules, aspects_per_edu)
        graph = self._calculate_edges_weight(graph)
        graph = self._delete_temporary_support_info(graph)

        aspect = None

        if conceptnet_io:
            # add relation from conceptnet
            for doc in documents_info.values():
                try:
                    cnio = doc['aspect_concepts']['conceptnet_io']
                    for aspect, concepts in cnio.iteritems():
                        log.info(aspect)
                        for concept in concepts:
                            graph.add_edge(concept['start'], concept['end'],
                                           relation_type=concept['relation'])
                except KeyError:
                    log.info('Aspect not in ConceptNet.io: {}'.format(aspect))

        page_ranks = self._calculate_page_ranks(graph)

        # todo
        # Gerani's dir-moi(a) = sentiment^2 -> importance of node/aspect

        return graph, page_ranks
