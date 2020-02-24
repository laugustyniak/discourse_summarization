import logging
from collections import namedtuple
from typing import List

from nltk.tree import Tree

EDURelation = namedtuple('EDURelation', 'edu1 edu2 relation_type gerani')

loger = logging.getLogger()


class EDUTreeRulesExtractor:
    def __init__(self, tree: Tree, weight_type: List[str] = None, only_hierarchical_relations: bool = True):
        """
        Extracting rules from RST tress.

        rules - dictionary of rules extracted from Discourse Trees, key is
            document id, value list of rules for tree
        tree - Discourse Tree
        left_child_parent - parent of actually analyzed left leaf
        right_child_parent - parent of actually analyzed right leaf

        :param only_hierarchical_relations: bool
            Do we want only hierarchical type of relations (especially from
            RST), True as default
        :param weight_type - list of weights calculated for discourse tree and
                their relations.
                'gerani' - weight according to publication  Gerani, S., Mehdad,
                    Y., Carenini, G., Ng, R. T., & Nejat, B. (2014).
                    Abstractive Summarization of Product Reviews Using
                    Discourse Structure. Emnlp, 1602-1613.
        """
        if weight_type is None:
            weight_type = ['gerani']
        self.rules: List[EDURelation] = []
        self.tree: Tree = tree
        self.left_child_parent = None
        self.left_leaf = None
        self.right_child_parent = None
        self.right_leaf = None
        self.weight_type = [w.lower() for w in weight_type]
        self.only_hierarchical_relations: bool = only_hierarchical_relations

    def extract(self) -> List[EDURelation]:
        try:
            if len(self.tree) == 0:
                # empty rules list
                return self.rules
            self._process_tree(self.tree)
        # TODO: fix AttributeError: 'int' object has no attribute 'height' in gerani calculation
        except AttributeError as e:
            loger.info(f'Error with parsing tree: {self.tree}')
            loger.info(f'Error: {str(e)}')
            return []
        return self.rules

    def _process_tree(self, tree):
        # there are max two childred for each subtree
        if len(tree) > 1:
            self.left_child_parent = tree
            self.right_child_parent = tree
        else:
            self.left_child_parent = tree

        for child in tree:
            # check if child is leaf or subtree
            if isinstance(child, Tree):
                # go into subtree
                self._process_tree(child)
            # recursively until leaf
            else:
                # leaf, parent/current subtree
                self._traverse_parent(child, tree)

    def _traverse_parent(self, leaf, parent):
        """ we reached leaf and want to parse sibling of leaf """
        # leaf = child
        # todo: check if none or sth other
        if parent is not None:
            self.right_child_parent = parent
            for child in parent:
                if child != leaf:
                    self._make_rules(leaf, child)

            # go up in the tree
            try:
                self.relation = parent.label()
                self._traverse_parent(leaf, parent.parent)
            except AttributeError:
                pass

    def _make_rules(self, leaf_left, tree):
        # if anything other than Tree we got leaf level
        if not isinstance(tree, Tree):
            self.left_leaf = leaf_left
            self.right_leaf = tree
            relation = self.rst_relation_type()
            # relation name, nucleus/satellite, nucleus/satellite
            rel_name, nuc_sat_1, nuc_sat_2 = self.get_nucleus_satellite_and_relation_type(relation)
            if self.only_hierarchical_relations and not self.check_hierarchical_rst_relation(nuc_sat_1, nuc_sat_2):
                return
            else:
                if nuc_sat_1 == 'N':
                    # [N][S] or [N][N]
                    self.rules.append(
                        EDURelation(
                            self.right_leaf,
                            self.left_leaf,
                            rel_name,
                            self.calculate_gerani_weight()
                        )
                    )
                else:
                    # [S][N]
                    self.rules.append(
                        EDURelation(
                            self.left_leaf,
                            self.right_leaf,
                            rel_name,
                            self.calculate_gerani_weight()
                        )
                    )
        # do deeper into tree
        else:
            for child in tree:
                self._make_rules(leaf_left, child)

    def calculate_gerani_weight(self):
        """
        Calculate weights for edu relations based on Gerani and
        Mehdad paper
        """
        if self.right_leaf is not None or self.left_leaf is not None:
            # calculate how many edus are between analyzed leafs,
            # leaf are integers hence we may substract them
            leaves = self.tree.leaves()
            n_edus_between_analyzed_edus = leaves.index(self.right_leaf) - leaves.index(self.left_leaf)
            n_edus_in_tree = len(leaves)
            tree_height = self.tree.height()
            if self.left_child_parent.height() > self.right_child_parent.height():
                sub_tree_height = self.left_child_parent.height()
            else:
                sub_tree_height = self.right_child_parent.height()
            return round(1 - 0.5 * (
                    float(n_edus_between_analyzed_edus) / n_edus_in_tree)
                         - 0.5 * (float(sub_tree_height) / tree_height), 2)
        return 0

    def rst_relation_type(self):
        """ Find common nearest parent and take relation from higher parse tree """
        if not isinstance(self.left_child_parent, Tree):
            return self.right_child_parent.label()
        elif self.left_child_parent.height() > self.right_child_parent.height():
            return self.left_child_parent.label()
        else:
            return self.right_child_parent.label()

    def get_nucleus_satellite_and_relation_type(self, relation):
        """
        Get nucleus/satellite pairs from RST relation and relation type name.

        :param relation: string
            Relation name with nucleus and satellite
        :return: tuple
            Nucleus or satellite indicators, ex. ('Elaboration', 'N', 'S')
        """
        relation_name = relation.split('[')[0]
        relation = relation[-5:-1].replace('][', '')
        return relation_name, relation[0], relation[1]

    def check_hierarchical_rst_relation(self, rel_1, rel_2):
        """
        Check if relation between analyzed parts of RST tree is hierarchical,
        relation between Nucleus and Satellite or no such as Nucleus-Nucleus
        relation

        :param rel_1: str
            Nucleus 'N' or satellite 'S' indicator
        :param rel_2: str
            Nucleus 'N' or satellite 'S' indicator

        :return bool
            True if hierarchical, False otherwise
        """
        return False if rel_1 == rel_2 else True
