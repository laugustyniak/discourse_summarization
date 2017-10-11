import unittest
import sys

import networkx as nx

from aspects.aspects.aspects_graph_builder import AspectsGraphBuilder
from aspects.utilities.data_paths import sample_tree_177, sample_tree_189

sys.path.append("../edu_dependency_parser/src")

from trees.parse_tree import ParseTree


class AspectGraphBuilderTest(unittest.TestCase):

    def _set_rst_rules_document_info(self):
        self.rest_rules = [(u'a movie', u'a film', u'Elaboration')]
        self.document_info = [{'EDUs': [0, 1, 2],
                               'accepted_edus': [2],
                               'aspect_concepts': {
                                   'conceptnet_io': {
                                       {u'film': [{'end': u'a film',
                                                   'end-lang': u'en',
                                                   'relation': u'IsA',
                                                   'start': u'a movie',
                                                   'start-lang': u'en',
                                                   'weight': 3.46},
                                                  ]}},
                                   'sentic': {}},
                               'aspect_keywords': {
                                   'rake': [(u'perfect', 1.0),
                                            (u'pretty', 1.0)]},
                               'aspects': [],
                               'sentiment': [0, 0, 1]},
                              {'EDUs': [3, 4],
                               'accepted_edus': [3, 4],
                               'aspect_concepts': {'conceptnet_io': {},
                                                   'sentic': {}},
                               'aspect_keywords': {
                                   'rake': [(u'browsing', 1.0)]},
                               'aspects': [],
                               'sentiment': [-1, 1]},
                              ]
        self.graph = nx.DiGraph()
        self.graph.add_edge(u'a movie', u'a film', relation_type=u'Elaboration')
        self.graph.add_edge(u'a movie', u'a film', relation_type=u'Background')
        self.graph.add_edge(u'a movie', u'a film', relation_type=u'IsA')

    def _set_parse_tree(self):
        self.parse_tree = ParseTree('same-unit[N][N]',
                                    [ParseTree('Elaboration[N][S]',
                                               [513, 514]),
                                     ParseTree('Elaboration[N][S]',
                                               [515,
                                                ParseTree(
                                                    'Joint[N][N]',
                                                    [516,
                                                     ParseTree(
                                                         'Elaboration[N][S]',
                                                         [517, 518])])])])

    def test_build_arrg_graph_rst_conceptnnet_io(self):
        self._set_rst_rules_document_info()
        aspects_graph_builder = AspectsGraphBuilder()
        graph_obtained = None

        # self.assertEqual(aspects_obtained, aspects_expected)

    def test_rst_relation_type(self):
        self._set_parse_tree()

    def build_exemplary_arrg_graph_sample_tree_177(self):
