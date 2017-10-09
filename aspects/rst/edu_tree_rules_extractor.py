# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import sys
import os

sys.path.append(os.getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree


class EDUTreeRulesExtractor(object):
    def __init__(self, weight_type=['gerani']):
        """
        Extracting rules from RST tress.

        rules - list of rules extracted from Discourse Trees
        tree - Discource Tree
        accepted_edus - list of edu ids that consist of aspect
        left_child_parent - parent of actually analyzed left leaf
        right_child_parent - parent of actually analyzed right leaf

        :param weight_type - list of weights calculated for discourse tree and
        their relations
        """
        self.rules = []
        self.tree = None
        self.accepted_edus = None
        self.left_child_parent = None
        self.left_leaf = None
        self.right_child_parent = None
        self.right_leaf = None
        self.weight_type = [w.lower() for w in weight_type]
        self.weight_mapping = {'gerani': self.gerani,
                               'relation_type': self.rst_relation_type}

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
            if tree in self.accepted_edus and leaf_left in self.accepted_edus:
                self.left_leaf = leaf_left
                self.right_leaf = tree
                weights = {k: v for k, v in self.weight_mapping.iteritems() if
                           k in self.weight_type}
                self.rules.append((self.left_leaf, self.right_leaf,
                                   self.rst_relation_type(), weights))
        # do deeper into tree
        else:
            for index, child in enumerate(tree):
                self._make_rules(leaf_left, child)

    def extract(self, tree, accepted_edus):
        self.accepted_edus = accepted_edus
        self.tree = tree
        self._process_tree(tree)

        return self.rules

        # INFO: reguly są determinowane przez kolejnosc
        # odwiedzania węzłów przy preprocessingu
        #   Jeśli olejemy wartość relacji, reguły mogą
        # być budowane od danego numeru do konca numerków:
        #   k -> k+1, k -> k+2, ... k -> n

    def gerani(self):
        """ Calculate weights for edu relations based on
        Gerani and Mehdad paper """
        # calculate how many edus are between analyzed leaf,
        # leaf are integers hence we may substrct them
        n_edus_between_analyzed_edus = self.right_leaf - self.left_leaf - 1
        n_edus_in_tree = len(self.tree.leaves())

        tree_height = self.tree.height()
        if self.left_child_parent.height() > self.right_child_parent.height():
            sub_tree_height = self.left_child_parent.height()
        else:
            sub_tree_height = self.right_child_parent.height()
        return 1 - 0.5 * (
        float(n_edus_between_analyzed_edus) / n_edus_in_tree) - 0.5 * (
        float(tree_height) / sub_tree_height)

    def rst_relation_type(self):
        """
        Find common nearest parent and take relation from heigher
        parse tree. In additin, check if relation if
        """
        if self.left_child_parent.height() > self.right_child_parent.height():
            return self.left_child_parent.node
        else:
            return self.right_child_parent.node
