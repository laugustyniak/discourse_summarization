import os
import sys
from collections import namedtuple

sys.path.append(os.getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree

Relation = namedtuple('Relation', 'edu1 edu2 relation_type gerani')


class EDUTreeRulesExtractor(object):
    def __init__(self, weight_type=None, only_hierarchical_relations=True):
        """
        Extracting rules from RST tress.

        rules - dictionaty of rules extracted from Discourse Trees, key is
            document id, value lsit of rules for tree
        tree - Discource Tree
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
        self.rules = []
        self.tree = None
        self.left_child_parent = None
        self.left_leaf = None
        self.right_child_parent = None
        self.right_leaf = None
        self.weight_type = [w.lower() for w in weight_type]
        self.only_hierarchical_relations = only_hierarchical_relations
        self.doc_id = None

    def _process_tree(self, tree):
        for child_index, child in enumerate(tree):
            # go to the subtree left child_index = 0, or right child_index = 1
            if not child_index:
                self.left_child_parent = tree
            else:
                self.right_child_parent = tree
            # check if child is leaf or subtree
            if isinstance(child, ParseTree):
                # go into subtree
                self._process_tree(child)
            # recursively until leaf
            else:
                # leaf, parent/current subtree, child_index of leaf in the tree
                self._traverse_parent(child, tree, child_index)

    def _traverse_parent(self, leaf, parent, child_index):
        """ we reached leaf and want to parse sibling of leaf """
        # leaf = child
        if parent is not None:
            # get sibling of leaf or in false you got the same leaf
            for index in range(child_index + 1, len(parent)):
                self.right_child_parent = parent
                self._make_rules(leaf, parent[index])

            # go up in the tree
            if parent.parent is not None:
                self.relation = parent.node
                self._traverse_parent(leaf, parent.parent, parent.parent_index)

    def _make_rules(self, leaf_left, tree):
        # if int we got leaf level
        if isinstance(tree, int):
            self.left_leaf = leaf_left
            self.right_leaf = tree
            relation = self.rst_relation_type()
            # relation name, nucleus/satellite, nucleus/satellite
            relation_name, nuc_sat_1, nuc_sat_2 = self.get_nucleus_satellite_and_relation_type(relation)
            if self.only_hierarchical_relations and not self.check_hierarchical_rst_relation(nuc_sat_1, nuc_sat_2):
                return
            else:
                if nuc_sat_1 == 'N':
                    # [N][S] or [N][N]
                    self.rules.append(Relation(
                        self.right_leaf, self.left_leaf, relation_name, self.calculate_gerani_weight()))
                else:
                    # [S][N]
                    self.rules.append(
                        Relation(self.left_leaf, self.right_leaf, relation_name, self.calculate_gerani_weight()))
        # do deeper into tree
        else:
            for index, child in enumerate(tree):
                self._make_rules(leaf_left, child)

    def extract(self, tree):
        """
        Extract RST relation with BFS approach.

        :param tree: ParseTree object
            RST Tree that will be parsed and relation will be extracted from it

        :return: rules list
            List of namedtuples with nodes/edu and metadata (relations and weights)
        """
        self.rules = []
        self.tree = tree
        self._process_tree(tree)

        return self.rules

    def calculate_gerani_weight(self):
        """ Calculate weights for edu relations based on Gerani and Mehdad paper """
        if self.right_leaf is not None or self.left_leaf is not None:
            # calculate how many edus are between analyzed leaf,
            # leaf are integers hence we may substract them
            n_edus_between_analyzed_edus = self.right_leaf - self.left_leaf - 1
            n_edus_in_tree = len(self.tree.leaves())
            tree_height = self.tree.height()
            if self.left_child_parent.height() > self.right_child_parent.height():
                sub_tree_height = self.left_child_parent.height()
            else:
                sub_tree_height = self.right_child_parent.height()
            return round(1 - 0.5 * (float(n_edus_between_analyzed_edus) / n_edus_in_tree)
                         - 0.5 * (float(sub_tree_height) / tree_height), 2)
        return 0

    def rst_relation_type(self):
        """
        Find common nearest parent and take relation from heigher
        parse tree
        """
        if self.left_child_parent.height() > self.right_child_parent.height():
            return self.left_child_parent.node
        else:
            return self.right_child_parent.node

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
