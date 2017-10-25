import unittest

import networkx as nx

from aspects.results_analysis.gerani_graph_analysis import get_dir_moi_for_node, \
    calculate_moi_by_gerani


class GeraniGraphAnalysisTest(unittest.TestCase):
    def test_get_dir_moi_for_node(self):
        aspects_per_edu = [(559, [u'phone']),  # test added manually
                           (560, [u'apple']),
                           (561, [u'apple']),
                           (562,
                            [u'store clerk', u'apple', u'teenager', u'advice']),
                           (563, []),
                           ]
        documents_info = {189: {'EDUs': [559, 560, 561, 562, 563],
                                'accepted_edus': [559, 560, 561, 562, 563],
                                'sentiment': {559: -1, 560: 1, 561: 1, 562: -1},
                                # the rest of document info is skipped
                                }
                          }
        graph = nx.DiGraph()
        graph.add_edge('screen', 'phone')
        graph.add_edge('speaker', 'apple')

        graph_expected = get_dir_moi_for_node(graph, aspects_per_edu,
                                              documents_info)
        attribs = nx.get_node_attributes(graph_expected, 'dir_moi')
        self.assertEqual(attribs['phone'], 1)
        self.assertEqual(attribs['apple'], 3)

    def test_calculate_moi_by_gerani(self):
        graph = nx.DiGraph()
        graph.add_edge('screen', 'phone')
        graph.node['phone']['dir_moi'] = 1
        graph.add_edge('speaker', 'apple')
        graph.node['apple']['dir_moi'] = 2

        graph_expected, aspect_moi = calculate_moi_by_gerani(graph)
        attribs = nx.get_node_attributes(graph_expected, 'moi')
        self.assertEqual(attribs['phone'], 0.6622808011387423)
        self.assertEqual(attribs['apple'], 1.1622808011387424)
        print(aspect_moi)
        self.assertEqual(aspect_moi, {'phone': 0.6622808011387423,
                                      'screen': 0.5877191988612577,
                                      'speaker': 0.5877191988612577,
                                      'apple': 1.1622808011387424})
