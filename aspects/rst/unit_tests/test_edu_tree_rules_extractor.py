import sys
import unittest

from aspects.data_io.serializer import Serializer
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.rst.edu_tree_rules_extractor import Relation
from aspects.utilities import settings

sys.path.append("../../../edu_dependency_parser/src")

from trees.parse_tree import ParseTree


class AspectExtractionTest(unittest.TestCase):
    def setUp(self):
        self.serializer = Serializer()
        self.rules_extractor = EDUTreeRulesExtractor()
        self.link_tree = self.serializer.load(settings.SAMPLE_TREE_177.as_posix())

    def _multi_aspect_per_edu_tree(self):
        self.link_tree = self.serializer.load(settings.SAMPLE_TREE_189.as_posix())

    def test_load_serialized_tree(self):
        self.assertEqual(isinstance(self.link_tree, ParseTree), True)
        self.assertEqual(self.link_tree.height(), 5)

    def test_tree_parsing_and_get_empty_rules(self):
        rules = self.rules_extractor.extract(self.link_tree, [], 0)
        self.assertEqual(len(rules), 0)

    def test_tree_parsing_and_get_rules_hierarchical(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'],
                                                only_hierarchical_relations=True)
        rules = rules_extractor.extract(self.link_tree,
                                        [513, 514, 515, 516, 517],
                                        1)
        expected_rules = {1: [Relation(edu1=514, edu2=513, relation_type='Elaboration', gerani=0.8),
                              Relation(edu1=516, edu2=515, relation_type='Elaboration', gerani=0.6),
                              Relation(edu1=517, edu2=515, relation_type='Elaboration', gerani=0.52)]}
        self.assertEqual(rules, expected_rules)

    def test_tree_parsing_and_get_rules_all(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'],
                                                only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.link_tree,
                                        [513, 514, 515, 516, 517],
                                        1)
        expected_rules = {1: [Relation(edu1=514, edu2=513, relation_type='Elaboration', gerani=0.8),
                              Relation(edu1=515, edu2=513, relation_type='same-unit', gerani=0.42),
                              Relation(edu1=516, edu2=513, relation_type='same-unit', gerani=0.33),
                              Relation(edu1=517, edu2=513, relation_type='same-unit', gerani=0.25),
                              Relation(edu1=515, edu2=514, relation_type='same-unit', gerani=0.5),
                              Relation(edu1=516, edu2=514, relation_type='same-unit', gerani=0.42),
                              Relation(edu1=517, edu2=514, relation_type='same-unit', gerani=0.33),
                              Relation(edu1=516, edu2=515, relation_type='Elaboration', gerani=0.6),
                              Relation(edu1=517, edu2=515, relation_type='Elaboration', gerani=0.52),
                              Relation(edu1=517, edu2=516, relation_type='Joint', gerani=0.7)]}
        self.assertEqual(rules, expected_rules)

    def test_get_nucleus_and_satellite(self):
        nucleus_satellite_pairs = {'same-unit[N][N]': ('same-unit', 'N', 'N'),
                                   'Elaboration[N][S]': (
                                       'Elaboration', 'N', 'S'),
                                   'Joint[N][N]': ('Joint', 'N', 'N')}
        for rel, ns in nucleus_satellite_pairs.iteritems():
            self.assertEqual(
                self.rules_extractor.get_nucleus_satellite_and_relation_type(
                    rel), ns)

    def test_check_if_hierarchical_rst_relation(self):
        to_check_hierarchicality = {('N', 'N'): False,
                                    ('N', 'S'): True,
                                    ('S', 'N'): True,
                                    }
        for (rel_1, rel_2), expected in to_check_hierarchicality.iteritems():
            self.assertEqual(
                self.rules_extractor.check_hierarchical_rst_relation(rel_1,
                                                                     rel_2),
                expected)

    def test_bfs_for_several_aspects_in_one_edu(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'],
                                                only_hierarchical_relations=True)
        rules = rules_extractor.extract(self.link_tree,
                                        [513, 514, 515, 516, 517],
                                        1)
        expected_rules = {1: [Relation(edu1=514, edu2=513, relation_type='Elaboration', gerani=0.8),
                              Relation(edu1=516, edu2=515, relation_type='Elaboration', gerani=0.6),
                              Relation(edu1=517, edu2=515, relation_type='Elaboration', gerani=0.52)]}
        self.assertEqual(rules, expected_rules)

    def test_multi_aspects_per_edu(self):
        self._multi_aspect_per_edu_tree()
        rules_extractor = EDUTreeRulesExtractor(
            only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.link_tree,
                                        [559, 560, 561, 562, 563],
                                        doc_id=1)
        expected_rules = {1: [Relation(edu1=560, edu2=559, relation_type='Elaboration', gerani=0.63),
                              Relation(edu1=561, edu2=559, relation_type='Elaboration', gerani=0.53),
                              Relation(edu1=562, edu2=559, relation_type='same-unit', gerani=0.3),
                              Relation(edu1=563, edu2=559, relation_type='same-unit', gerani=0.2),
                              Relation(edu1=561, edu2=560, relation_type='Elaboration', gerani=0.75),
                              Relation(edu1=562, edu2=560, relation_type='same-unit', gerani=0.4),
                              Relation(edu1=563, edu2=560, relation_type='same-unit', gerani=0.3),
                              Relation(edu1=562, edu2=561, relation_type='same-unit', gerani=0.5),
                              Relation(edu1=563, edu2=561, relation_type='same-unit', gerani=0.4),
                              Relation(edu1=563, edu2=562, relation_type='Elaboration', gerani=0.75)]}
        self.assertEqual(rules, expected_rules)
