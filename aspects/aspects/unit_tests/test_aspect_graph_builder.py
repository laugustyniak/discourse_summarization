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
        # todo check document info structure
        # fixme udpate rules structure
        self.rst_rules = [(u'a movie', u'a film', u'Elaboration')]
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
                               'aspects': {2: [u'film']},
                               'sentiment': {0: 0, 1: 0, 2: 1}},
                              {'EDUs': [3, 4],
                               'accepted_edus': [3, 4],
                               'aspect_concepts': {'conceptnet_io': {},
                                                   'sentic': {}},
                               'aspect_keywords': {
                                   'rake': [(u'browsing', 1.0)]},
                               'aspects': {},
                               'sentiment': {3: -1, 4: 1}},
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
        aspects_per_edu = [(559, [u'phone']),  # test added manually
                           (560, [u'apple']),
                           (561, []),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, []),
                           ]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562], 1)
        documents_info = {189: {'EDUs': [559, 560, 561, 562, 563],
                                'accepted_edus': [559, 560, 561, 562, 563],
                                'aspect_concepts':
                                    {'conceptnet_io': {
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
                                                    'weight': 2.8284271},
                                                   ]}
                                    },
                                # the rest of document info is skipped
                                }
                          }
        graph, page_rank = aspects_graph_builder.build(rules, documents_info,
                                                       True)
        self.assertEqual(len(rules), 1)
        self.assertGreaterEqual(len(graph.nodes()), 4)
        self.assertGreaterEqual(len(graph.edges()), 3)
        self.assertEqual(graph['apple']['phone']['relation_type'],
                         'Elaboration')
        self.assertEqual(graph['object']['thing']['relation_type'], 'IsA')
        self.assertEqual(graph['thing']['stuff']['relation_type'], 'Synonym')

        # self.assertEqual(aspects_obtained, aspects_expected)

    def test_rst_relation_type(self):
        self._set_parse_tree()

    def test_build_exemplary_arrg_graph_sample_tree_189(self):
        self._setup_link_parse_tree_189()
        aspects_per_edu = [(559, [u'test']),  # test added manually
                           (560, [u'thing']),
                           (561, []),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, [])]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562], 1)
        graph, page_rank = aspects_graph_builder.build(rules, None, False)
        self.assertEqual(len(rules), 1)
        self.assertGreaterEqual(len(graph.nodes()), 1)
        self.assertGreaterEqual(len(graph.edges()), 1)
        self.assertEqual(graph['thing']['test']['relation_type'], 'Elaboration')

    def test_build_exemplary_arrg_graph_sample_tree_189_multiaspects(self):
        self._setup_link_parse_tree_189()
        aspects_per_edu = [(559, [u'test', u'test2']),  # test added manually
                           (560, [u'thing', u'test2']),
                           (561, []),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, [])]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562], 1)
        graph, page_rank = aspects_graph_builder.build(rules,
                                                       conceptnet_io=False)
        self.assertEqual(len(rules), 1)
        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 4)
        self.assertEqual(graph['thing']['test']['relation_type'], 'Elaboration')

    def test_filter_only_max_weight(self):
        rules = {1: [(514, 513, 'Elaboration', -0.25),
                     (514, 513, 'Elaboration', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 124124.38),
                     (516, 515, 'Elaboration', 0.29)],
                 }
        aspects_per_edu = [(513, [u'513']),  # test added manually
                           (514, [u'514']),
                           (515, [u'515']),
                           (516, [u'516'])]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_obtained = aspects_graph_builder.filter_only_max_gerani_weight_multi_rules(
            rules)
        rules_expected = {
            1: [('514', '513', 'Elaboration', 124124.38),
                ('516', '515', 'Elaboration', 0.29)]}
        self.assertEqual(rules_obtained, rules_expected)

    def test_filter_only_max_weight_different_relation_type(self):
        rules = {1: [(514, 513, 'Elaboration', -0.25),
                     (514, 513, 'Contrast', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Contrast', 124124.38),
                     (516, 515, 'Elaboration', 0.29)],
                 }
        aspects_per_edu = [(513, [u'513']),  # test added manually
                           (514, [u'514']),
                           (515, [u'515']),
                           (516, [u'516'])]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_obtained = aspects_graph_builder.filter_only_max_gerani_weight_multi_rules(
            rules)
        rules_expected = {
            1: [('514', '513', 'Contrast', 124124.38),
                ('514', '513', 'Elaboration', 1.38),
                ('516', '515', 'Elaboration', 0.29),
                ]}
        self.assertEqual(rules_obtained, rules_expected)

    def test_filter_only_max_weight_multiaspects(self):
        rules = {1: [(514, 513, 'Elaboration', -0.25),
                     (514, 513, 'Elaboration', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (517, 513, 'Elaboration', 94.38),
                     (516, 515, 'Elaboration', 0.29),
                     (516, 513, 'Elaboration', 0.2),
                     ]
                 }
        aspects_per_edu = [(513, [u'513', u'test13']),  # test added manually
                           (514, [u'514', u'test14']),
                           (515, [u'515']),
                           (516, [u'516']),
                           (517, [u'test17']),
                           ]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_obtained = aspects_graph_builder.filter_only_max_gerani_weight_multi_rules(
            rules)
        rules_expected = {
            1: [(u'514', u'513', u'Elaboration', 1.38),
                (u'514', u'test13', u'Elaboration', 1.38),
                (u'516', u'513', u'Elaboration', 0.2),
                (u'516', u'515', u'Elaboration', 0.29),
                (u'516', u'test13', u'Elaboration', 0.2),
                (u'test14', u'513', u'Elaboration', 1.38),
                (u'test14', u'test13', u'Elaboration', 1.38),
                (u'test17', u'513', u'Elaboration', 94.38),
                (u'test17', u'test13', u'Elaboration', 94.38),
                ]
        }
        self.assertEqual(rules_obtained, rules_expected)

    def test_get_maximum_confidence_rule_per_doc(self):
        rules = {1: [(514, 513, 'Elaboration', -0.25),
                     (514, 513, 'Elaboration', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (517, 513, 'Elaboration', 94.38),
                     (516, 515, 'Elaboration', 0.29),
                     (516, 513, 'Elaboration', 0.2),
                     ],
                 2: [(514, 513, 'same-unit', -0.25),
                     (514, 513, 'same-unit', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (517, 513, 'Elaboration', 94.38),
                     (516, 515, 'Elaboration', 0.29),
                     (516, 513, 'Elaboration', 0.2),
                     ],
                 }
        aspects_per_edu = [(513, [u'513', u'test13']),  # test added manually
                           (514, [u'514', u'test14']),
                           (515, [u'515']),
                           (516, [u'516']),
                           (517, [u'test17']),
                           ]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_obtained = aspects_graph_builder.get_maximum_confidence_rule_per_doc(
            rules, top_n_rules=1)
        rules_expected = {1: [(u'test14', u'test13', 'Elaboration')],
                          2: [(u'514', u'test13', 'Elaboration')]}
        self.assertEqual(rules_obtained, rules_expected)
