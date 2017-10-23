import logging

from collections import OrderedDict, defaultdict, Counter, namedtuple
from itertools import groupby
from operator import itemgetter

import networkx as nx
import operator

log = logging.getLogger(__name__)

Relation = namedtuple('Relation', 'aspect1 aspect2 relation_type gerani')


class AspectsGraphBuilder(object):
    def __init__(self, aspects_per_edu=None):
        """

        Parameters
        ----------
        aspects_per_edu : list
            List of rules aspect, aspect, relation and weights.

        """
        if aspects_per_edu is None:
            aspects_per_edu = []
        self.aspects_per_edu = dict(aspects_per_edu)

    def build(self, rules, documents_info=None,
              conceptnet_io=False, filter_gerani=False):
        """
        Build aspect(EDU)-aspect(EDU) network based on RST and ConceptNet
        relation.

        Parameters
        ----------
        filter_gerani: bool
            Do we want to get only max weight if there are more than one
            rule with the same node_1, node_2 and relation type in processed
            RST Tree? False as default.

        rules: dict
            Dictionary with document id and list of rules that will be used to
            create aspect-aspect graph, list elements: (node_1, node_2,
            relation type, weight).

        documents_info: dict
            Dictionary with information about each edu.

        conceptnet_io: bool
            Do we use ConcetNet.io relation in graph?

        Returns
        -------
        graph: networkx.Graph
            Graph with aspect-aspect relations

        page_rank: networkx.PageRank
            PageRank counted for aspect-aspect graph.

        """
        if documents_info is None:
            documents_info = {}

        if filter_gerani:
            rules = self.filter_gerani(rules)
        graph = self._build_aspects_graph(rules)
        graph = self._calculate_edges_support(graph)
        # graph = self._delete_nodes_attribute(graph, 'support')

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

        return graph, page_ranks

    def _add_node_to_graph(self, graph, node):
        """
        Add node with attributes to graph.

        :param graph: networkx.Graph
            Aspect-aspect graph.

        :param node: str
            node name

        :return:
            graph: networkx.Graph
                Aspect-aspect graph with extended nodes attributes.
        """
        if graph.has_node(node):
            graph.node[node]['counter'] += 1
        else:
            graph.add_node(node, {'counter': 1})

        return graph

    def _add_edge_to_graph(self, graph, node_left, node_right,
                           relation_type='None'):
        if graph.has_edge(node_left, node_right):
            graph[node_left][node_right]['counter'] += 1
        else:
            graph.add_edge(node_left, node_right)
            graph[node_left][node_right]['counter'] = 1

        graph[node_left][node_right]['relation_type'] = relation_type

        return graph

    def _build_aspects_graph(self, rules):
        """
        Build graph based on list of tuples with apsects ids.

        :param rules: list
            (aspect_id_1, aspect_id_2, relation, weight_dict)

        :return: networkx.DiGraph()
            Graph with aspect-aspect relation.
        """
        graph = nx.DiGraph()

        for doc_id, rules in rules.iteritems():
            for rule in rules:
                log.debug('Rule: {}'.format(rule))
                left_node, right_node, relation, weigths = rule

                for aspect_left, aspect_right in self.aspects_iterator(
                        left_node, right_node):
                    graph = self._add_node_to_graph(graph, aspect_left)
                    graph = self._add_node_to_graph(graph, aspect_right)
                    graph = self._add_edge_to_graph(graph, aspect_left,
                                                    aspect_right,
                                                    relation_type=relation)
        return graph

    def _calculate_edges_support(self, graph):
        """
        Calculate confidence/weight of each node/aspect
        Parameters
        ----------
        graph : networkx.Graph
            Graph of aspect-aspect relation ARRG.

        Returns
        -------
        graph : networkx.Graph
            Graph of aspect-aspect relation ARRG with calculated confidence.

        """
        attribute = 'counter'
        for edge in graph.edges():
            edge_support = graph[edge[0]][edge[1]][attribute]
            first_node_support = graph.node[edge[0]][attribute]
            graph[edge[0]][edge[1]]['support'] = \
                edge_support / float(first_node_support)
            del graph[edge[0]][edge[1]][attribute]

        return graph

    def _delete_nodes_attribute(self, graph, attibute):
        """
        Remove atttibute from all nodes.

        Parameters
        ----------
        graph : networkx.Graph
            Graph of aspect-aspect relation ARRG.

        attibute : str
            Name of attibute to be removed from all nodes.

        Returns
        -------
        graph : networkx.Graph
            Graph of aspect-aspect relation ARRG without attribute in nodes.
        """
        for node in graph.nodes():
            del graph.node[node][attibute]

        return graph

    def _calculate_page_ranks(self, graph):
        """
        Calculate Page Rank for ARRG.

        Parameters
        ----------
        graph : networkx.Graph
            Graph of aspect-aspect relation ARRG.

        Returns
        -------
        page_ranks : OrderedDict
            PAge Rank values for ARRG.

        """
        page_ranks = nx.pagerank(graph)
        page_ranks = OrderedDict(sorted(page_ranks.items(), key=itemgetter(1),
                                        reverse=True))

        return page_ranks

    def aspects_iterator(self, edu_id_1, edu_id_2):
        """
        Generator for aspect pairs of provided edu id pairs.

        Parameters
        ----------
        edu_id_1 : int
            EDU ID
        edu_id_2 : int
            EDU ID

        Returns
        -------
        tuple
            (aspect, aspect)
        """
        try:
            for aspect_left in self.aspects_per_edu[edu_id_1]:
                for aspect_right in self.aspects_per_edu[edu_id_2]:
                    yield (aspect_left, aspect_right)
        except KeyError as err:
            log.info('Lack of aspect: {}'.format(str(err)))

    def filter_only_max_gerani_weight_multi_rules_per_doc(self, rules):
        """
        Filter rules that are dupliates and got only these with max weight.

        Parameters
        ----------
        rules : dict
            Dictionary of document id and list of rules. Relation tuple
            Relation(aspect_right, aspect, relation, gerani_weight)

        Returns
        -------
        rules : dict
            Dictionary of document id and list of rules.
        """
        rules_filtered = defaultdict(list)
        for doc_id, rules_list in rules.iteritems():
            for rule in rules_list:
                log.debug('Rule: {}'.format(rule))
                left_node, right_node, relation, gerani_weight = rule
                for aspect_left, aspect_right in self.aspects_iterator(
                        left_node, right_node):
                    rules_filtered[doc_id].append(Relation(aspect_left,
                                                           aspect_right,
                                                           relation,
                                                           gerani_weight))
        for doc_id, rls in rules_filtered.iteritems():
            rules_filtered[doc_id] = [max(v, key=lambda x: x[3])
                                      for
                                      g, v in
                                      groupby(sorted(rls), key=lambda x: x[:3])]
        return rules_filtered

    def filter_only_max_gerani_weight_multi_rules(self, rules):
        """
        Filter rules that are dupliates and got only these with max weight.

        Parameters
        ----------
        rules : dict
            Dictionary of document id and list of rules. Relation tuple
            Relation(aspect_right, aspect, relation, gerani_weight)

        Returns
        -------
        rules : dict
            Dictionary of document id and list of rules. The only key is -1
            indicating that only one relation of each combination
            (aspect, aspect, relation) with the highest gerani weight has been
            choosen.

        """
        rules_filtered = []
        for doc_id, rules_list in rules.iteritems():
            for rule in rules_list:
                log.debug('Rule: {}'.format(rule))
                left_node, right_node, relation, gerani_weigth = rule
                for aspect_left, aspect_right in self.aspects_iterator(
                        left_node, right_node):
                    rules_filtered.append(Relation(aspect_left, aspect_right,
                                                   relation, gerani_weigth))
        log.info('#{} rules in #{} documents'.format(len(rules_filtered),
                                                     len(rules)))
        rules_with_heighest_gerani_weight = {
            -1: [max(v, key=lambda x: x[3]) for g, v in
                 groupby(sorted(rules_filtered), key=lambda x: x[:3])]}
        log.info('#{} rules after filtering'.format(
            len(rules_with_heighest_gerani_weight[-1])))
        return rules_with_heighest_gerani_weight

    def filter_maximum_confidence_rule_per_doc(self, rules, top_n_rules=1):
        """
        Filter rules by its confidence,

        Parameters
        ----------
        rules : dict
            Dictionary of document id and list of rules.

        top_n_rules : int
            How many rules with highest confidence do we want to extract? 1 as
            default.

        Returns
        -------
        rules_filtered : defaultdict
            Dictionary of document id and list of rules.


        """
        rules_filtered = defaultdict(list)
        for doc_id, rules_list in rules.iteritems():
            rules_confidence = defaultdict(list)
            for left_node, right_node, relation, gerani in rules_list:
                for aspect_left, aspect_right in self.aspects_iterator(
                        int(left_node), int(right_node)):
                    rules_confidence[doc_id].append(Relation(aspect_left,
                                                             aspect_right,
                                                             relation, gerani))
            # count aspect, aspect, relation tuples
            relation_counter = Counter(
                [x[:3] for x in rules_confidence[doc_id]])
            # count confidence, dict {(asp1, asp2, rel): confidence}
            rules_confidence = {rel: float(freq) / len(rules_confidence) for
                                rel, freq in
                                relation_counter.iteritems()}
            print(
                'Doc id: {} and rules; {}'.format(doc_id, rules_confidence))
            # sort by confidence
            rules_confidence = sorted(rules_confidence,
                                      key=operator.itemgetter(1), reverse=True)
            # filter top n rules by confidence
            rules_filtered[doc_id].extend(rules_confidence[:top_n_rules])
        return rules_filtered

    def filter_gerani(self, rules):
        rules = self.filter_only_max_gerani_weight_multi_rules_per_doc(rules)
        rules = self.filter_maximum_confidence_rule_per_doc(rules)
        rules = self.filter_only_max_gerani_weight_multi_rules(rules)
        return rules
