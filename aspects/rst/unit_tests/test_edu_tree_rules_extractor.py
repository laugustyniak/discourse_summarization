import unittest
import sys

from aspects.io.serializer import Serializer
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor

sys.path.append("../../../edu_dependency_parser/src")

from trees.parse_tree import ParseTree


class AspectExtractionTest(unittest.TestCase):
    def setUp(self):
        self.serializer = Serializer()
        self.rules_extractor = EDUTreeRulesExtractor()
        self.link_tree = self.serializer.load('../sample_trees/177')

    def test_load_serialized_tree(self):
        self.assertEqual(isinstance(self.link_tree, ParseTree), True)
        self.assertEqual(self.link_tree.height(), 5)

    def test_tree_parsing_and_get_empty_rules(self):
        rules = self.rules_extractor.extract(self.link_tree, [])
        self.assertEqual(len(rules), 0)

    def test_tree_parsing_and_get_rules(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'])
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
