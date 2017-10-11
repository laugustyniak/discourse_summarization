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
        rules = self.rules_extractor.extract(self.link_tree, [])
        self.assertEqual(len(rules), 0)

    def test_tree_parsing_and_get_rules_hierarchical(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'],
                                                only_hierarchical_relations=True)
        rules = rules_extractor.extract(self.link_tree,
                                        [513, 514, 515, 516, 517])
        expected_rules = [(513, 514, 'Elaboration[N][S]', {'gerani': -0.25}),
                          (515, 516, 'Elaboration[N][S]', {'gerani': 0.375}),
                          (515, 517, 'Elaboration[N][S]',
                           {'gerani': 0.29166666666666663})]
        self.assertEqual(rules, expected_rules)

    def test_tree_parsing_and_get_rules_all(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'],
                                                only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.link_tree,
                                        [513, 514, 515, 516, 517])
        expected_rules = [(513, 514, 'Elaboration[N][S]', {'gerani': -0.25}),
                          (513, 515, 'same-unit[N][N]',
                           {'gerani': 0.41666666666666663}), (
                              513, 516, 'same-unit[N][N]',
                              {'gerani': 0.33333333333333337}), (
                              513, 517, 'same-unit[N][N]',
                              {'gerani': 0.25}),
                          (514, 515, 'same-unit[N][N]',
                           {'gerani': 0.5}),
                          (514, 516, 'same-unit[N][N]',
                           {'gerani': 0.41666666666666663}), (
                              514, 517, 'same-unit[N][N]',
                              {'gerani': 0.33333333333333337}), (
                              515, 516, 'Elaboration[N][S]',
                              {'gerani': 0.375}),
                          (515, 517, 'Elaboration[N][S]',
                           {'gerani': 0.29166666666666663}), (
                              516, 517, 'Joint[N][N]',
                              {'gerani': 0.16666666666666663})]
        self.assertEqual(rules, expected_rules)

    def test_get_nucleus_and_satellite(self):
        nucleus_satellite_pairs = {'same-unit[N][N]': ('N', 'N'),
                                   'Elaboration[N][S]': ('N', 'S'),
                                   'Joint[N][N]': ('N', 'N')}
        for rel, ns in nucleus_satellite_pairs.iteritems():
            self.assertEqual(
                self.rules_extractor.get_nucleus_and_satellite(rel), ns)

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
                                        [513, 514, 515, 516, 517])
        expected_rules = [(513, 514, 'Elaboration[N][S]', {'gerani': -0.25}),
                          (515, 516, 'Elaboration[N][S]', {'gerani': 0.375}),
                          (515, 517, 'Elaboration[N][S]',
                           {'gerani': 0.29166666666666663})]
        self.assertEqual(rules, expected_rules)

    def test_multi_aspects_per_edu(self):
        self._multi_aspect_per_edu_tree()
        rules_extractor = EDUTreeRulesExtractor(
            only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.link_tree,
                                        [559, 560, 561, 562, 563])
        expected_rules = [
            (559, 560, 'Elaboration[N][S]', {'gerani': 0.33333333333333337}),
            (559, 561, 'Elaboration[N][S]', {'gerani': 0.2333333333333334}),
            (559, 562, 'same-unit[N][N]', {'gerani': 0.30000000000000004}),
            (559, 563, 'same-unit[N][N]', {'gerani': 0.19999999999999996}),
            (560, 561, 'Elaboration[N][S]', {'gerani': 0.0}),
            (560, 562, 'same-unit[N][N]', {'gerani': 0.4}),
            (560, 563, 'same-unit[N][N]', {'gerani': 0.30000000000000004}),
            (561, 562, 'same-unit[N][N]', {'gerani': 0.5}),
            (561, 563, 'same-unit[N][N]', {'gerani': 0.4}),
            (562, 563, 'Elaboration[N][S]', {'gerani': 0.0})]
        self.assertEqual(rules, expected_rules)
