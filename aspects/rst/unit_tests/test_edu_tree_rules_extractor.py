import unittest

from nltk.tree import Tree

from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor, EDURelation
from aspects.utilities import settings


class AspectExtractionTest(unittest.TestCase):

    def _load_tree(self, discourse_tree: str) -> Tree:
        return Tree.fromstring(
            discourse_tree,
            leaf_pattern=settings.DISCOURSE_TREE_LEAF_PATTERN,
            remove_empty_top_bracketing=True
        )

    def _with_simple_discourse_tree(self):
        self.discourse_tree = self._load_tree(settings.SAMPLE_TREE_177.open('r').read())

    def _with_celery_text_discourse_tree(self):
        self.discourse_tree = self._load_tree(settings.SAMPLE_TREE_1.open('r').read())

    def _multi_aspect_per_edu_tree(self):
        self.discourse_tree = self._load_tree(settings.SAMPLE_TREE_189.open('r').read())

    def test_load_serialized_tree(self):
        self._with_simple_discourse_tree()
        self.rules_extractor = EDUTreeRulesExtractor(self.discourse_tree, only_hierarchical_relations=True)
        rules = self.rules_extractor.extract()
        self.assertEqual(len(rules), 5)

    def test_extract_rules_from_simple_celery_discourse_tree_hierarchical_only_relations(self):
        self._with_celery_text_discourse_tree()
        self.rules_extractor = EDUTreeRulesExtractor(self.discourse_tree, only_hierarchical_relations=False)
        rules = self.rules_extractor.extract()
        self.assertEqual(len(rules), 6)

    def test_tree_parsing_and_get_rules_hierarchical(self):
        rules_extractor = EDUTreeRulesExtractor(
            weight_type=['gerani'], only_hierarchical_relations=True)
        rules = rules_extractor.extract(self.discourse_tree,
                                        [513, 514, 515, 516, 517],
                                        1)
        expected_rules = {1: [EDURelation(edu1=514, edu2=513, relation_type='Elaboration', gerani=0.8),
                              EDURelation(edu1=516, edu2=515, relation_type='Elaboration', gerani=0.6),
                              EDURelation(edu1=517, edu2=515, relation_type='Elaboration', gerani=0.52)]}
        self.assertEqual(rules, expected_rules)

    def test_tree_parsing_and_get_rules_all(self):
        rules_extractor = EDUTreeRulesExtractor(weight_type=['gerani'],
                                                only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.discourse_tree,
                                        [513, 514, 515, 516, 517],
                                        1)
        expected_rules = {1: [EDURelation(edu1=514, edu2=513, relation_type='Elaboration', gerani=0.8),
                              EDURelation(edu1=515, edu2=513, relation_type='same-unit', gerani=0.42),
                              EDURelation(edu1=516, edu2=513, relation_type='same-unit', gerani=0.33),
                              EDURelation(edu1=517, edu2=513, relation_type='same-unit', gerani=0.25),
                              EDURelation(edu1=515, edu2=514, relation_type='same-unit', gerani=0.5),
                              EDURelation(edu1=516, edu2=514, relation_type='same-unit', gerani=0.42),
                              EDURelation(edu1=517, edu2=514, relation_type='same-unit', gerani=0.33),
                              EDURelation(edu1=516, edu2=515, relation_type='Elaboration', gerani=0.6),
                              EDURelation(edu1=517, edu2=515, relation_type='Elaboration', gerani=0.52),
                              EDURelation(edu1=517, edu2=516, relation_type='Joint', gerani=0.7)]}
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
        rules = rules_extractor.extract(self.discourse_tree,
                                        [513, 514, 515, 516, 517],
                                        1)
        expected_rules = {1: [EDURelation(edu1=514, edu2=513, relation_type='Elaboration', gerani=0.8),
                              EDURelation(edu1=516, edu2=515, relation_type='Elaboration', gerani=0.6),
                              EDURelation(edu1=517, edu2=515, relation_type='Elaboration', gerani=0.52)]}
        self.assertEqual(rules, expected_rules)

    def test_multi_aspects_per_edu(self):
        self._multi_aspect_per_edu_tree()
        rules_extractor = EDUTreeRulesExtractor(
            only_hierarchical_relations=False)
        rules = rules_extractor.extract(self.discourse_tree,
                                        [559, 560, 561, 562, 563],
                                        doc_id=1)
        expected_rules = {1: [EDURelation(edu1=560, edu2=559, relation_type='Elaboration', gerani=0.63),
                              EDURelation(edu1=561, edu2=559, relation_type='Elaboration', gerani=0.53),
                              EDURelation(edu1=562, edu2=559, relation_type='same-unit', gerani=0.3),
                              EDURelation(edu1=563, edu2=559, relation_type='same-unit', gerani=0.2),
                              EDURelation(edu1=561, edu2=560, relation_type='Elaboration', gerani=0.75),
                              EDURelation(edu1=562, edu2=560, relation_type='same-unit', gerani=0.4),
                              EDURelation(edu1=563, edu2=560, relation_type='same-unit', gerani=0.3),
                              EDURelation(edu1=562, edu2=561, relation_type='same-unit', gerani=0.5),
                              EDURelation(edu1=563, edu2=561, relation_type='same-unit', gerani=0.4),
                              EDURelation(edu1=563, edu2=562, relation_type='Elaboration', gerani=0.75)]}
        self.assertEqual(rules, expected_rules)
