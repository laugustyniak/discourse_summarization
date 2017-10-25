import logging

from collections import OrderedDict, defaultdict, Counter, namedtuple
from itertools import groupby
from operator import itemgetter

import networkx as nx
import operator

from aspects.utilities.transformations import flatten_list

log = logging.getLogger(__name__)

RelationAspects = namedtuple('Relation',
                             'aspect1 aspect2 relation_type gerani_weight')


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
        graph = self.build_aspects_graph(rules)

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

        page_ranks = self.calculate_page_ranks(graph, weight='gerani_weight')
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
                           relation_type='None', gerani_weight=0):
        if graph.has_edge(node_left, node_right):
            graph[node_left][node_right]['counter'] += 1
        else:
            graph.add_edge(node_left, node_right)
            graph[node_left][node_right]['counter'] = 1

        graph[node_left][node_right]['relation_type'] = relation_type
        graph[node_left][node_right]['gerani_weight'] = gerani_weight

        return graph

    def build_aspects_graph(self, rules):
        """
        Build graph based on list of tuples with apsects ids.

        :param rules: list
            (aspect_id_1, aspect_id_2, relation, weight_dict)

        :return: networkx.DiGraph()
            Graph with aspect-aspect relation.
        """
        graph = nx.DiGraph()
        for doc_id, rules_list in rules.iteritems():
            for rule in rules_list:
                log.debug('Rule: {}'.format(rule))

                if isinstance(rule, RelationAspects):
                    aspect_left, aspect_right, relation, gerani_weigth = rule
                    graph = self._add_node_to_graph(graph, aspect_left)
                    graph = self._add_node_to_graph(graph, aspect_right)
                    graph = self._add_edge_to_graph(graph, aspect_left,
                                                    aspect_right,
                                                    relation_type=relation,
                                                    gerani_weight=gerani_weigth)
                else:
                    left_node, right_node, relation, gerani_weigth = rule
                    for aspect_left, aspect_right in self.aspects_iterator(
                            left_node, right_node):
                        graph = self._add_node_to_graph(graph, aspect_left)
                        graph = self._add_node_to_graph(graph, aspect_right)
                        graph = self._add_edge_to_graph(graph, aspect_left,
                                                        aspect_right,
                                                        relation_type=relation,
                                                        gerani_weight=gerani_weigth)
        return graph

    def _calculate_edges_support(self, graph):
        """
        Calculate confidence/weight of each node/aspect.

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

    def calculate_page_ranks(self, graph, weight='weight'):
        """
        Calculate Page Rank for ARRG.

        Parameters
        ----------
        graph : networkx.Graph
            Graph of aspect-aspect relation ARRG.

        weight : str, optional
            Name of edge attribute that consists of weight for an endge. it is
            used to calculate Weighted version of Page Rank.

        Returns
        -------
        page_ranks : OrderedDict
            PAge Rank values for ARRG.

        """
        page_ranks = nx.pagerank(graph, weight=weight)
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
            raise KeyError(str(err))

    def filter_gerani(self, rules):
        """
        Filter rules by its confidence,

        Parameters
        ----------
        rules : dict
            Dictionary of document id and list of rules. Relation tuple
            Relation(aspect_right, aspect, relation, gerani_weight)

        Returns
        -------
        rules_filtered : defaultdict
            Dictionary of document id (-1 as indicatior for all documents)
            and list of rules/relations between aspects with their maximum
            gerani weights.

            Examples
            {-1: [Relation(aspect1=u'screen', aspect2=u'phone',
                                        relation_type='Elaboration',
                                        gerani_weight=1.38),
                               Relation(aspect1=u'speaker', aspect2=u'sound',
                                        relation_type='Elaboration',
                                        gerani_weight=0.29)]}
        """
        rule_per_doc = defaultdict(list)
        for doc_id, rules_list in rules.iteritems():
            rules_filtered = []
            for rule in rules_list:
                log.debug('Rule: {}'.format(rule))
                left_node, right_node, relation, gerani_weight = rule
                for aspect_left, aspect_right in self.aspects_iterator(
                        left_node, right_node):
                    rules_filtered.append(RelationAspects(aspect_left,
                                                          aspect_right,
                                                          relation,
                                                          gerani_weight))
            if len(rules_filtered):
                relation_counter = Counter([x[:3] for x in rules_filtered])
                rule_confidence = sorted(relation_counter,
                                         key=operator.itemgetter(1),
                                         reverse=True)[0]
                rules_confidence = [r for r in rules_filtered
                                    if rule_confidence == r[:3]]
                rule_per_doc[doc_id].extend(
                    [max(v, key=lambda rel: rel.gerani_weight) for g, v in
                     groupby(sorted(rules_confidence),
                             key=lambda rel: rel[:3])])
            else:
                log.info('Empty rule list for document {}'.format(doc_id))
        relations_list = [
            (group + (sum([rel.gerani_weight for rel in relations]),))
            for group, relations in
            groupby(sorted(flatten_list(rule_per_doc.values())),
                    key=lambda rel: rel[:3])]
        # map relations into namedtuples
        relations_list = [RelationAspects(a1, a2, r, w) for
                          a1, a2, r, w in relations_list]
        rules = {-1: relations_list}
        return rules
