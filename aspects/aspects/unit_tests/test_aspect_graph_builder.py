import unittest
import sys

import networkx as nx

from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.aspects.aspects_graph_builder import AspectsGraphBuilder
from aspects.io.serializer import Serializer
from aspects.utilities.data_paths import sample_tree_177, sample_tree_189

sys.path.append("../edu_dependency_parser/src")

from trees.parse_tree import ParseTree


class AspectGraphBuilderTest(unittest.TestCase):
    def setUp(self):
        self.serializer = Serializer()

    def _setup_link_parse_tree_177(self):
        self.link_tree = self.serializer.load(sample_tree_177)

    def _setup_link_parse_tree_189(self):
        self.link_tree = self.serializer.load(sample_tree_189)

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
        self._setup_link_parse_tree_189()
        aspects_graph_builder = AspectsGraphBuilder()
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562])
        aspects_per_edu = [(559, [u'phone']),  # test added manually
                           (560, [u'apple']),
                           (561, []),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, [])]
        documents_info = {189:
                              {'EDUs': [559, 560, 561, 562, 563],
                               'accepted_edus': [559, 560, 561, 562, 563],
                               'aspect_concepts':
                                   {
                                       559: {'conceptnet_io': {}, 'sentic': {}},
                                       560: {'conceptnet_io': {
                                           u'thing': [{'end': u'thing',
                                                       'end-lang': u'en',
                                                       'relation': u'IsA',
                                                       'start': u'object',
                                                       'start-lang': u'en',
                                                       'weight': 2.8284271},
                                                      {'end': u'stuff',
                                                       'end-lang': u'en',
                                                       'relation': u'Synonym',
                                                       'start': u'thing',
                                                       'start-lang': u'en',
                                                       'weight': 2.8284271}
                                                      ]}
                                       }
                                   }
                               }
                          }
        graph, page_rank = aspects_graph_builder.build(rules, aspects_per_edu,
                                                       documents_info, True)
        self.assertEqual(len(rules), 1)
        self.assertGreaterEqual(len(graph.nodes()), 4)
        self.assertGreaterEqual(len(graph.edges()), 3)
        self.assertEqual(graph['phone']['apple']['relation_type'],
                         'Elaboration')
        self.assertEqual(graph['object']['thing']['relation_type'], 'IsA')
        self.assertEqual(graph['thing']['stuff']['relation_type'], 'Synonym')

        # self.assertEqual(aspects_obtained, aspects_expected)

    def test_rst_relation_type(self):
        self._set_parse_tree()

    def test_build_exemplary_arrg_graph_sample_tree_189(self):
        self._setup_link_parse_tree_189()
        aspects_graph_builder = AspectsGraphBuilder()
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562])
        aspects_per_edu = [(559, [u'test']),  # test added manually
                           (560, [u'thing']),
                           (561, []),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, [])]
        graph, page_rank = aspects_graph_builder.build(rules, aspects_per_edu,
                                                       None, False)
        self.assertEqual(len(rules), 1)
        self.assertGreaterEqual(len(graph.nodes()), 1)
        self.assertGreaterEqual(len(graph.edges()), 1)
        self.assertEqual(graph['test']['thing']['relation_type'], 'Elaboration')

    def test_build_exemplary_arrg_graph_sample_tree_189_multiaspects(self):
        self._setup_link_parse_tree_189()
        aspects_graph_builder = AspectsGraphBuilder()
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562])
        aspects_per_edu = [(559, [u'test', u'test2']),  # test added manually
                           (560, [u'thing', u'test2']),
                           (561, []),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, [])]
        graph, page_rank = aspects_graph_builder.build(rules, aspects_per_edu,
                                                       None, False)
        self.assertEqual(len(rules), 1)
        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 4)
        self.assertEqual(graph['test']['thing']['relation_type'], 'Elaboration')
