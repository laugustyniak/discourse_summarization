import unittest
import sys

from aspects.io.serializer import Serializer
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.utilities.data_paths import sample_tree_177, sample_tree_189

sys.path.append("../../../edu_dependency_parser/src")

from trees.parse_tree import ParseTree


class AspectExtractionTest(unittest.TestCase):
    def setUp(self):
        self.serializer = Serializer()
        self.rules_extractor = EDUTreeRulesExtractor()
        self.link_tree = self.serializer.load(sample_tree_177)

    def _multi_aspect_per_edu_tree(self):
        self.link_tree = self.serializer.load(sample_tree_189)

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
        expected_rules = {1: [(514, 513, 'Elaboration', {'gerani': -0.25}),
                              (516, 515, 'Elaboration', {'gerani': 0.38}),
                              (517, 515, 'Elaboration', {'gerani': 0.29})]}
        self.assertEqual(rules, expected_rules)

    def test_tree_parsing_and_get_rules_all(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'],
                                                only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.link_tree,
                                        [513, 514, 515, 516, 517],
                                        1)
        expected_rules = {1: [(514, 513, 'Elaboration', {'gerani': -0.25}),
                              (515, 513, 'same-unit', {'gerani': 0.42}),
                              (516, 513, 'same-unit', {'gerani': 0.33}),
                              (517, 513, 'same-unit', {'gerani': 0.25}),
                              (515, 514, 'same-unit', {'gerani': 0.5}),
                              (516, 514, 'same-unit', {'gerani': 0.42}),
                              (517, 514, 'same-unit', {'gerani': 0.33}),
                              (516, 515, 'Elaboration', {'gerani': 0.38}),
                              (517, 515, 'Elaboration', {'gerani': 0.29}),
                              (517, 516, 'Joint', {'gerani': 0.17})]}
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
        expected_rules = {1: [(514, 513, 'Elaboration', {'gerani': -0.25}),
                              (516, 515, 'Elaboration', {'gerani': 0.38}),
                              (517, 515, 'Elaboration', {'gerani': 0.29})]}
        self.assertEqual(rules, expected_rules)

    def test_multi_aspects_per_edu(self):
        self._multi_aspect_per_edu_tree()
        rules_extractor = EDUTreeRulesExtractor(
            only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.link_tree,
                                        [559, 560, 561, 562, 563],
                                        doc_id=1)
        expected_rules = {1: [
            (560, 559, 'Elaboration', {'gerani': 0.33}),
            (561, 559, 'Elaboration', {'gerani': 0.23}),
            (562, 559, 'same-unit', {'gerani': 0.3}),
            (563, 559, 'same-unit', {'gerani': 0.2}),
            (561, 560, 'Elaboration', {'gerani': 0.0}),
            (562, 560, 'same-unit', {'gerani': 0.4}),
            (563, 560, 'same-unit', {'gerani': 0.3}),
            (562, 561, 'same-unit', {'gerani': 0.5}),
            (563, 561, 'same-unit', {'gerani': 0.4}),
            (563, 562, 'Elaboration', {'gerani': 0.0})]}
        self.assertEqual(rules, expected_rules)
