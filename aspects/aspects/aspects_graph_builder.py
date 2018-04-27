import logging
import operator
from collections import OrderedDict, defaultdict, Counter, namedtuple
from itertools import groupby
from operator import itemgetter

import networkx as nx

from aspects.analysis.gerani_graph_analysis import get_dir_moi_for_node, calculate_moi_by_gerani
from aspects.io.serializer import Serializer
from aspects.utilities.transformations import flatten_list

log = logging.getLogger(__name__)

RelationAspects = namedtuple('Relation', 'aspect1 aspect2 relation_type gerani_weight')


class AspectsGraphBuilder(object):
    def __init__(self, aspects_per_edu=None, alpha_gerani=0.5, with_cycles_between_aspects=False):
        """

        Parameters
        ----------
        aspects_per_edu : list
            List of rules aspect, aspect, relation and weights.

        alpha_gerani : float
            Alpha parameter for moi function. 0.5 as default.

        with_cycles_between_aspects : bool
            Do we want to have cycles for aspect, in arrg there could be aspect1-aspect1 relation or not. False by
            default.

        """
        self.with_cycles_between_aspects = with_cycles_between_aspects
        self.alpha_gerani = alpha_gerani
        if aspects_per_edu is None:
            aspects_per_edu = []
        self.aspects_per_edu = dict(aspects_per_edu)
        self.serializer = Serializer()

    def build(self, rules, docs_info=None, conceptnet_io=False, filter_gerani=False, aht_gerani=False,
              aspect_graph_path=None):
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

        aht_gerani : bool
            Do we want to follow Gerani's approach and calculate mai function?
            ARRG would be updated accordingly is True. False as default.

        docs_info: dict
            Dictionary with information about each edu.

        conceptnet_io: bool
            Do we use ConcetNet.io relation in graph?

        aspect_graph_path : str
            Path to save temporar arrg.

        Returns
        -------
        graph: networkx.MultiDiGraph
            Graph with aspect-aspect relations

        page_rank: networkx.PageRank
            PageRank counted for aspect-aspect graph.

        """
        if docs_info is None:
            docs_info = {}
        if filter_gerani:
            rules = self.filter_gerani(rules)
        graph = self.build_aspects_graph(rules)

        aspect = None
        # fixme please fix fucking iteration via aspects_concept dicts
        # started rewriting for data frames in other branch
        if conceptnet_io:
            # add relation from conceptnet
            for doc_id, doc_info in docs_info.iteritems():
                try:
                    for _, concepts_all in doc_info['aspect_concepts'].iteritems():
                        for aspect, concepts in concepts_all['conceptnet_io'].iteritems():
                            log.info(aspect)
                            for concept in concepts:
                                graph.add_edge(concept['start'], concept['end'], relation_type=concept['relation'])
                except KeyError:
                    log.info('Aspect not in ConceptNet.io: {}'.format(aspect))

        if aspect_graph_path is not None:
            log.info('Save ARRG graph based on rules only (with conceptnets relations!)')
            nx.write_gexf(graph, aspect_graph_path + '_based_on_rules_only.gexf')

        graph = get_dir_moi_for_node(graph, self.aspects_per_edu, docs_info)
        graph, page_ranks = calculate_moi_by_gerani(graph, self.alpha_gerani)
        if aht_gerani:
            graph = self.arrg_to_aht(graph=graph, weight='gerani_weight')
        else:
            page_ranks = self.calculate_page_ranks(graph, weight='gerani_weight')
        return graph, page_ranks

    def build_aspects_graph(self, rules):
        """
        Build graph based on list of tuples with apsects ids.

        Parameters
        ----------
        rules : list
            List of tuples (aspect_id_1, aspect_id_2, relation, gerani_weight)
            or namedtuple as RelationAspects()

        Returns
        -------
        graph : networkx.MultiDiGraph
            Graph with aspect-aspect relation.
        """
        graph = nx.MultiDiGraph()
        for doc_id, rules_list in rules.iteritems():
            for rule in rules_list:
                log.debug('Rule: {}'.format(rule))
                if isinstance(rule, RelationAspects):
                    aspect_left, aspect_right, relation, gerani_weight = rule
                    graph = self.add_aspects_to_graph(graph, aspect_left, aspect_right, relation, gerani_weight)
                else:
                    left_node, right_node, relation, gerani_weight = rule
                    for aspect_left, aspect_right in self.aspects_iterator(
                            left_node, right_node):
                        graph = self.add_aspects_to_graph(graph, aspect_left, aspect_right, relation, gerani_weight)

        return graph

    def add_aspects_to_graph(self, graph, aspect_left, aspect_right, relation, gerani_weight):
        if self.with_cycles_between_aspects or aspect_left != aspect_right:
            log.info('Skip rule: {}-{}'.format(aspect_left, aspect_right))
            graph.add_edge(aspect_left, aspect_right, relation_type=relation, gerani_weight=gerani_weight)
        return graph

    def calculate_page_ranks(self, graph, weight='weight'):
        """
        Calculate Page Rank for ARRG.

        Parameters
        ----------
        graph : networkx.MultiDiGraph
            Graph of aspect-aspect relation ARRG.

        weight : str, optional
            Name of edge attribute that consists of weight for an endge. it is
            used to calculate Weighted version of Page Rank.

        Returns
        -------
        page_ranks : OrderedDict
            Page Rank values for ARRG.

        """
        log.info('Weighted Page Rank calculation starts here.')
        page_ranks = nx.pagerank_scipy(graph, weight=weight)
        log.info('Weighted Page Rank calculation ended.')
        page_ranks = OrderedDict(sorted(page_ranks.items(), key=itemgetter(1), reverse=True))
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

    def merge_multiedges_in_arrg(self, graph, node_attrib_name='gerani_weight'):
        """
        Merge multiple edges between nodes into one relation and sum attribute weight.

        Merge the edges connecting two nodes and consider the sum of their weights as the weight of the merged graph.
        We also ignore the relation direction for the purpose of generating the tree.

        Parameters
        ----------
        graph : networkx.MultiDiGraph
            Aspect-aspect relation graph.

        node_attrib_name : str
            Name of node's attibute that will be summed up in merged relations.

        Returns
        -------
        graph_new : networkx.Graph()
            Aspect-aspect graph with maximum single relation between each node.

        """
        graph_new = nx.Graph()
        for u, v, data in graph.edges(data=True):
            w = data[node_attrib_name] if node_attrib_name in data else 1.0
            if graph_new.has_edge(u, v):
                graph_new[u][v][node_attrib_name] += w
            else:
                graph_new.add_edge(u, v, gerani_weight=w)
        return graph_new

    def arrg_to_aht(self, graph, weight):
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

        Returns
        -------

        """
        mst = nx.maximum_spanning_tree(graph, weight=weight)
        return mst
