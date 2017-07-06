# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import sys
import os

sys.path.append(os.getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree


class EDUTreeRulesExtractor(object):
    def __init__(self):
        self.rules = []
        self.accepted_edus = None

    def __process_tree(self, tree):
        for index, node in enumerate(tree):
            # go to the subtree
            if isinstance(node, ParseTree):
                self.__process_tree(node)
            # recursively until leaf
            else:
                # leaf, parent/current subtree, index of leaf in the tree
                self.__traverse_parent(node, tree, index)

    def __traverse_parent(self, leaf, parent, child_index):
        # leaf = child
        if parent is not None:
            for id in range(child_index + 1, len(parent)):
                self.__make_rules(leaf, parent[id])

            if parent.parent is not None:
                self.__traverse_parent(leaf, parent.parent, parent.parent_index)

    def __make_rules(self, base, tree):

        # int oznacza, że rekursywnie doszliśmy do liścia w drzewie pod aktualnym liście analizowanym
        if isinstance(tree, int):
            if tree in self.accepted_edus and base in self.accepted_edus:
                self.rules.append((base, tree))
        else:
            for index, subtree in enumerate(tree):
                self.__make_rules(base, subtree)

    def extract(self, tree, accepted_edus):
        self.accepted_edus = accepted_edus
        self.__process_tree(tree)

        return self.rules

        # INFO: reguly są determinowane przez kolejnosc odwiedzania węzłów przy preprocessingu
        #   Jeśli olejemy wartość relacji, reguły mogą być budowane od danego numeru do konca numerków:
        #   k -> k+1, k -> k+2, ... k -> n
