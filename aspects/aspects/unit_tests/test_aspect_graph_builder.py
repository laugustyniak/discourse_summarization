import unittest
from collections import OrderedDict

import networkx as nx
from nltk.tree import Tree

from aspects.aspects.aspects_graph_builder import Aspect2AspectGraph
from aspects.data_io import serializer
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.utilities import settings


class AspectGraphBuilderTest(unittest.TestCase):

    def _setup_link_parse_tree_177(self):
        self.link_tree = serializer.load(settings.SAMPLE_TREE_177.as_posix())

    def _setup_link_parse_tree_189(self):
        self.link_tree = serializer.load(settings.SAMPLE_TREE_189.as_posix())

    def _set_rst_rules_document_info(self):
        self.rules = {1: [(514, 513, 'Elaboration', 0.25),
                          (514, 513, 'Elaboration', 1.38),
                          (514, 513, 'Elaboration', 0.38),
                          ],
                      2: [(514, 513, 'same-unit', 0.25),
                          (514, 513, 'same-unit', 0.38),
                          (514, 513, 'Elaboration', 1.38),
                          (517, 513, 'Elaboration', 94.38),
                          (516, 515, 'Elaboration', 0.29),
                          (516, 513, 'Elaboration', 0.2),
                          ],
                      3: [],
                      }
        self.aspects_per_edu = [(513, [u'phone']),
                                (514, [u'screen', u'voicemail']),
                                (515, [u'sound']),
                                (516, [u'speaker']),
                                (517, [u'screen', u'speaker']),
                                ]
        self.docs_info = {1: {'sentiment': {513: 1, 514: -1, 515: 1, 517: -1}}}

    def _set_parse_tree(self):
        self.parse_tree = Tree('same-unit[N][N]',
                               [Tree('Elaboration[N][S]',
                                     [513, 514]),
                                Tree('Elaboration[N][S]',
                                     [515,
                                      Tree(
                                          'Joint[N][N]',
                                          [516,
                                           Tree(
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
        aspects_graph_builder = Aspect2AspectGraph(aspects_per_edu)
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562], 1)
        documents_info = {189: {'EDUs': [559, 560, 561, 562, 563],
                                'accepted_edus': [559, 560, 561, 562, 563],
                                'aspect_concepts':
                                    {189:
                                        {'conceptnet_io': {
                                            u'apple': [{'end': u'apple',
                                                        'end-lang': u'en',
                                                        'relation': u'IsA',
                                                        'start': u'object',
                                                        'start-lang': u'en',
                                                        'weight': 2.8284271},
                                                       {'end': u'stuff',
                                                        'end-lang': u'en',
                                                        'relation': u'Synonym',
                                                        'start': u'apple',
                                                        'start-lang': u'en',
                                                        'weight': 2.8284271},
                                                       ]}
                                        }
                                    },
                                'sentiment': {559: 1, 560: -1, 5562: 1},
                                # the rest of document info is skipped
                                }
                          }
        graph, page_rank = aspects_graph_builder.build(rules, documents_info, True)
        self.assertEqual(len(rules), 1)
        self.assertGreaterEqual(len(graph.nodes()), 4)
        self.assertGreaterEqual(len(graph.edges()), 3)
        attrib = nx.get_edge_attributes(graph, 'relation_type')
        self.assertEqual(attrib, {(u'object', u'apple', 0): u'IsA',
                                  (u'apple', u'stuff', 0): u'Synonym',
                                  (u'apple', u'phone', 0): 'Elaboration'})

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
        aspects_graph_builder = Aspect2AspectGraph(aspects_per_edu)
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562], 1)
        graph, page_rank = aspects_graph_builder.build(rules, None, False)
        self.assertEqual(len(rules), 1)
        self.assertGreaterEqual(len(graph.nodes()), 1)
        self.assertGreaterEqual(len(graph.edges()), 1)
        attrib = nx.get_edge_attributes(graph, 'relation_type')
        self.assertEqual(attrib, {(u'thing', u'test', 0): 'Elaboration'})

    def test_build_exemplary_arrg_graph_sample_tree_189_multiaspects(self):
        self._setup_link_parse_tree_189()
        aspects_per_edu = [(559, [u'test', u'test2']),  # test added manually
                           (560, [u'thing', u'test2']),
                           (561, []),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, [])]
        aspects_graph_builder = Aspect2AspectGraph(aspects_per_edu, with_cycles_between_aspects=True)
        rules_extractor = EDUTreeRulesExtractor()
        rules = rules_extractor.extract(self.link_tree, [559, 560, 562], 1)
        graph, page_rank = aspects_graph_builder.build(rules, conceptnet_io=False)
        self.assertEqual(len(rules), 1)
        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 4)
        attrib = nx.get_edge_attributes(graph, 'relation_type')
        self.assertEqual(attrib, {(u'thing', u'test', 0): 'Elaboration',
                                  (u'test2', u'test2', 0): 'Elaboration',
                                  (u'test2', u'test', 0): 'Elaboration',
                                  (u'thing', u'test2', 0): 'Elaboration'})

    def test_filter_gerani(self):
        rules = {1: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 2: [(514, 513, 'same-unit', 0.25),
                     (514, 513, 'same-unit', 0.38),
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
        aspects_graph_builder = Aspect2AspectGraph(aspects_per_edu)
        rules_obtained = aspects_graph_builder.filter_gerani(rules)
        rules_expected = {
            -1: [AspectsRelation(aspect1=u'screen', aspect2=u'phone', relation_type='Elaboration', gerani_weight=1.38),
                 AspectsRelation(aspect1=u'speaker', aspect2=u'sound', relation_type='Elaboration',
                                 gerani_weight=0.29)]}
        self.assertEqual(rules_obtained, rules_expected)

    def test_filter_gerani_sum_of_max_weights(self):
        rules = {1: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 2: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 10.8),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 3: [],
                 }
        aspects_per_edu = [(513, [u'phone']),
                           (514, [u'screen', u'voicemail']),
                           (515, [u'sound']),
                           (516, [u'speaker']),
                           (517, [u'screen', u'speaker']),
                           ]
        aspects_graph_builder = Aspect2AspectGraph(aspects_per_edu)
        rules_obtained = aspects_graph_builder.filter_gerani(rules)
        rules_expected = {
            -1: [AspectsRelation(aspect1=u'screen', aspect2=u'phone',
                                 relation_type='Elaboration',
                                 gerani_weight=12.18),
                 ]}
        self.assertEqual(rules_obtained, rules_expected)

    def test_build_without_conceptnet_multi_rules_filter_confidence_filter(
            self):
        rules = {1: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 2: [(514, 513, 'same-unit', 0.25),
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
        aspects_graph_builder = Aspect2AspectGraph(aspects_per_edu)
        graph, pagerank = aspects_graph_builder.build(rules, docs_info={}, conceptnet_io=False, filter_gerani=True)
        gerani_weight_attrib = nx.get_edge_attributes(graph, 'gerani_weight')
        relation_type_weight_attrib = nx.get_edge_attributes(graph, 'relation_type')
        self.assertTrue(isinstance(graph, nx.DiGraph))
        self.assertEqual(graph.number_of_nodes(), 4)
        gerani_weight_attrib_expected = {(u'screen', u'phone', 0): 1.38,
                                         (u'speaker', u'sound', 0): 0.29}
        relation_type_weight_attrib_expected = {(u'screen', u'phone', 0): 'Elaboration',
                                                (u'speaker', u'sound', 0): 'Elaboration'}
        self.assertEqual(gerani_weight_attrib, gerani_weight_attrib_expected)
        self.assertEqual(relation_type_weight_attrib, relation_type_weight_attrib_expected)

    def test_build_without_conceptnet_multi_rules_no_filter_confidence_filter(
            self):
        rules = {1: [(514, 513, 'Elaboration', 0.25),
                     (514, 513, 'Elaboration', 1.38),
                     (514, 513, 'Elaboration', 0.38),
                     ],
                 2: [(514, 513, 'same-unit', 0.25),
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
        aspects_graph_builder = Aspect2AspectGraph(aspects_per_edu, with_cycles_between_aspects=True)
        graph, pagerank = aspects_graph_builder.build(rules, docs_info={}, conceptnet_io=False, filter_gerani=False)
        self.assertTrue(isinstance(graph, nx.DiGraph))
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 16)
        self.assertEqual(graph['screen']['phone'],
                         {0: {'gerani_weight': 0.25, 'relation_type': 'Elaboration'},
                          1: {'gerani_weight': 1.38, 'relation_type': 'Elaboration'},
                          2: {'gerani_weight': 0.38, 'relation_type': 'Elaboration'},
                          3: {'gerani_weight': 0.25, 'relation_type': 'same-unit'},
                          4: {'gerani_weight': 0.38, 'relation_type': 'same-unit'},
                          5: {'gerani_weight': 1.38, 'relation_type': 'Elaboration'},
                          6: {'gerani_weight': 94.38, 'relation_type': 'Elaboration'},
                          }
                         )
        self.assertEqual(graph['speaker']['sound'],
                         {0: {'relation_type': 'Elaboration', 'gerani_weight': 0.29}}
                         )

        pagerank_expected = OrderedDict(
            [(u'phone', 0.4698552806383583), (u'sound', 0.1327942890741717), (u'screen', 0.1324501434291567),
             (u'speaker', 0.1324501434291567), (u'voicemail', 0.1324501434291567)])
        self.assertEqual(pagerank, pagerank_expected)

    def test_build_without_conceptnet_multi_rules_gerani_moi(self):
        self._set_rst_rules_document_info()
        aspects_graph_builder = Aspect2AspectGraph(self.aspects_per_edu, alpha_gerani=0.5)
        graph, pagerank = aspects_graph_builder.build(self.rules, docs_info=self.docs_info, conceptnet_io=False,
                                                      filter_gerani=True, aht_gerani=False)
        self.assertTrue(isinstance(graph, nx.DiGraph))
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(graph['screen']['phone'],
                         {0: {'relation_type': 'Elaboration', 'gerani_weight': 1.38}})
        self.assertEqual(graph['speaker']['sound'],
                         {0: {'relation_type': 'Elaboration', 'gerani_weight': 0.29}})

    def test_calculate_page_ranks(self):
        aspects_graph_builder = Aspect2AspectGraph()
        self._set_multigraph_arrg()
        pagerank = aspects_graph_builder.calculate_page_ranks(
            self.arrg_sample_graph, weight='gerani_weight')
        pagerank_expected = OrderedDict(
            [('phone', 0.46744998785621517), ('cellphone', 0.18167321669875405), ('screen', 0.17543839772251535),
             ('voicemail', 0.17543839772251535)])
        self.assertEqual(pagerank, pagerank_expected)

    def _set_multigraph_arrg(self):
        self.arrg_sample_graph = nx.DiGraph()
        self.arrg_sample_graph.add_edge('screen', 'phone', gerani_weight=10.38)
        self.arrg_sample_graph.add_edge('voicemail', 'phone', gerani_weight=100.38)
        self.arrg_sample_graph.add_edge('voicemail', 'cellphone', gerani_weight=4.38)

    def test_calculate_page_ranks_no_weight(self):
        aspects_graph_builder = Aspect2AspectGraph()
        self._set_multigraph_arrg()
        pagerank = aspects_graph_builder.calculate_page_ranks(self.arrg_sample_graph)
        pagerank_expected = OrderedDict(
            [('phone', 0.3991232045549693), ('cellphone', 0.25), ('screen', 0.17543839772251535),
             ('voicemail', 0.17543839772251535)])
        self.assertEqual(pagerank, pagerank_expected)

    def test_merge_multiedges_in_arrg(self):
        aspects_graph_builder = Aspect2AspectGraph()
        graph = nx.DiGraph()
        graph.add_edge('screen', 'phone', gerani_weight=10, relation='Elaboration')
        graph.add_edge('screen', 'phone', gerani_weight=10, relation='same-unit')
        graph.add_edge('voicemail', 'phone', gerani_weight=10, relation='Elaboration')
        graph.add_edge('voicemail', 'phone', gerani_weight=13, relation='same-unit')
        graph.add_edge('voicemail', 'phone', gerani_weight=12, relation='Contrast')
        graph.add_edge('signal', 'phone', gerani_weight=11, relation='Contrast')
        graph.add_edge('signal', 'phone', gerani_weight=9, relation='Elaboration')
        graph.add_edge('phone', 'signal', gerani_weight=17, relation='Elaboration')
        graph = aspects_graph_builder.merge_multiedges_in_arrg(graph)
        self.assertEqual(nx.get_edge_attributes(graph, name='gerani_weight'),
                         {('phone', 'voicemail'): 35, ('phone', 'signal'): 37, ('phone', 'screen'): 20})

    def test_arrg_to_aht(self):
        aspects_graph_builder = Aspect2AspectGraph()
        graph = nx.Graph()
        graph.add_edge('phone', 'screen', gerani_weight=20)
        graph.add_edge('phone', 'voicemail', gerani_weight=35)
        graph.add_edge('apple', 'voicemail', gerani_weight=22)
        graph.add_edge('apple', 'phone', gerani_weight=55)
        graph.add_edge('apple', 'screen', gerani_weight=55)
        graph.add_edge('apple', 'signal', gerani_weight=20)
        graph.add_edge('phone', 'signal', gerani_weight=37)
        graph.add_edge('phone', 'camera', gerani_weight=37)
        graph.add_edge('sound', 'apple', gerani_weight=7)
        graph.add_edge('sound', 'phone', gerani_weight=15)
        graph.add_edge('pixel', 'camera', gerani_weight=10)
        graph.add_edge('pixel', 'screen', gerani_weight=20)
        graph.add_edge('phone', 'screen', gerani_weight=7)
        mst_obtained = aspects_graph_builder.arrg_to_aht(graph, 'gerani_weight')
        self.assertEquals(len(mst_obtained.nodes()), len(graph.nodes()))
        self.assertLess(len(mst_obtained.edges()), len(graph.edges()))
        self.assertEqual(nx.get_edge_attributes(mst_obtained, name='gerani_weight'),
                         {('phone', 'voicemail'): 35,
                          ('apple', 'screen'): 55,
                          ('apple', 'phone'): 55,
                          ('signal', 'phone'): 37,
                          ('phone', 'camera'): 37,
                          ('screen', 'pixel'): 20,
                          ('sound', 'phone'): 15,
                          })

    def test_add_aspects_to_graph_not_equal_aspects(self):
        aspects_graph_builder = Aspect2AspectGraph(with_cycles_between_aspects=True)
        graph = nx.Graph()
        aspects_graph_builder.add_aspects_to_graph(graph, 'phone', 'battery', 'rel', 55)
        self.assertEquals(len(graph.nodes()), 2)
        self.assertEquals(len(graph.edges()), 1)

    def test_add_aspects_to_graph_equal_aspects(self):
        aspects_graph_builder = Aspect2AspectGraph(with_cycles_between_aspects=False)
        graph = nx.Graph()
        aspects_graph_builder.add_aspects_to_graph(graph, 'phone', 'phone', 'rel', 55)
        self.assertEquals(len(graph.nodes()), 0)
        self.assertEquals(len(graph.edges()), 0)
