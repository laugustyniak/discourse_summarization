import unittest
import sys
from collections import OrderedDict

import networkx as nx

from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.aspects.aspects_graph_builder import AspectsGraphBuilder, \
    RelationAspects
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

    def test_filter_gerani(self):
        rules = {1: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 2: [(514, 513, 'same-unit', -0.25),
                     (514, 513, 'same-unit', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (517, 513, 'Elaboration', 94.38),
                     (516, 515, 'Elaboration', 0.29),
                     (516, 513, 'Elaboration', 0.2),
                     ],
                 3: [],
                 }
        aspects_per_edu = [(513, [u'phone']),
                           (514, [u'screen', u'voicemail']),
                           (515, [u'sound']),
                           (516, [u'speaker']),
                           (517, [u'screen', u'speaker']),
                           ]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        rules_obtained = aspects_graph_builder.filter_gerani(rules)
        rules_expected = {
            -1: [RelationAspects(aspect1=u'screen', aspect2=u'phone',
                                 relation_type='Elaboration',
                                 gerani_weight=1.38),
                 RelationAspects(aspect1=u'speaker', aspect2=u'sound',
                                 relation_type='Elaboration',
                                 gerani_weight=0.29)]}
        self.assertEqual(rules_obtained, rules_expected)

    def test_build_without_conceptnet_multi_rules_filter_confidence_filter(
            self):
        rules = {1: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 2: [(514, 513, 'same-unit', -0.25),
                     (514, 513, 'same-unit', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (517, 513, 'Elaboration', 94.38),
                     (516, 515, 'Elaboration', 0.29),
                     (516, 513, 'Elaboration', 0.2),
                     ],
                 3: [],
                 }
        aspects_per_edu = [(513, [u'phone']),
                           (514, [u'screen', u'voicemail']),
                           (515, [u'sound']),
                           (516, [u'speaker']),
                           (517, [u'screen', u'speaker']),
                           ]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        graph, pagerank = aspects_graph_builder.build(rules,
                                                      documents_info={},
                                                      conceptnet_io=False,
                                                      filter_gerani=True,
                                                      )
        graph_expected = nx.DiGraph()
        graph_expected.add_edge('screen', 'phone')
        graph_expected['screen']['phone']['relation_type'] = 'Elaboration'
        graph_expected['screen']['phone']['gerani_weight'] = 1.38
        graph_expected.add_edge('speaker', 'sound')
        graph_expected['speaker']['sound']['relation_type'] = 'Elaboration'
        graph_expected['speaker']['sound']['gerani_weight'] = 0.29

        self.assertTrue(isinstance(graph, nx.DiGraph))
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph['screen']['phone']['relation_type'],
                         graph_expected['screen']['phone']['relation_type'])
        self.assertEqual(graph['screen']['phone']['gerani_weight'],
                         graph_expected['screen']['phone']['gerani_weight'])
        self.assertEqual(graph['speaker']['sound']['relation_type'],
                         graph_expected['speaker']['sound']['relation_type'])
        self.assertEqual(graph['speaker']['sound']['gerani_weight'],
                         graph_expected['speaker']['sound']['gerani_weight'])

    def test_build_without_conceptnet_multi_rules_no_filter_confidence_filter(
            self):
        rules = {1: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 2: [(514, 513, 'same-unit', -0.25),
                     (514, 513, 'same-unit', 0.38),
                     (514, 513, 'Elaboration', 1.38),
                     (517, 513, 'Elaboration', 94.38),
                     (516, 515, 'Elaboration', 0.29),
                     (516, 513, 'Elaboration', 0.2),
                     ],
                 3: [],
                 }
        aspects_per_edu = [(513, [u'phone']),
                           (514, [u'screen', u'voicemail']),
                           (515, [u'sound']),
                           (516, [u'speaker']),
                           (517, [u'screen', u'speaker']),
                           ]
        aspects_graph_builder = AspectsGraphBuilder(aspects_per_edu)
        graph, pagerank = aspects_graph_builder.build(rules,
                                                      documents_info={},
                                                      conceptnet_io=False,
                                                      filter_gerani=False,
                                                      )
        graph_expected = nx.DiGraph()
        graph_expected.add_edge('screen', 'phone')
        graph_expected['screen']['phone']['relation_type'] = 'Elaboration'
        graph_expected['screen']['phone']['gerani_weight'] = 1.38
        graph_expected.add_edge('speaker', 'sound')
        graph_expected['speaker']['sound']['relation_type'] = 'Elaboration'
        graph_expected['speaker']['sound']['gerani_weight'] = 0.29

        self.assertTrue(isinstance(graph, nx.DiGraph))
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 4)
        self.assertEqual(graph['screen']['phone'],
                         {'support': 1.0, 'gerani_weight': 94.38,
                          'relation_type': 'Elaboration'})

        pagerank_expected = OrderedDict(
            [(u'phone', 0.4139078791410039), (u'sound', 0.18874169057152615),
             (u'screen', 0.1324501434291567), (u'speaker', 0.1324501434291567),
             (u'voicemail', 0.1324501434291567)])
        self.assertEqual(pagerank, pagerank_expected)
