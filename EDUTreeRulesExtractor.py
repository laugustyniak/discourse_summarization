# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import sys
import os

sys.path.append(os.getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree


class EDUTreeRulesExtractor():
    def __processTree(self, tree):
        for index, node in enumerate(tree):
            if isinstance(node, ParseTree):
                self.__processTree(node)
            else:
                self.__traverseParent(node, tree, index)

    def __traverseParent(self, leaf, parent, childIndex):
        if parent is not None:
            for id in range(childIndex + 1, len(parent)):
                self.__makeRules(leaf, parent[id])

            if parent.parent is not None:
                self.__traverseParent(leaf, parent.parent, parent.parent_index)

    def __makeRules(self, base, tree):

        # int oznacza, że rekursywnie doszliśmy do liścia w drzewie pod aktualnym liście analizowanym
        if isinstance(tree, int):
            if tree in self.acceptedEDUs and base in self.acceptedEDUs:
                self.rules.append((base, tree))
        else:
            for index, subtree in enumerate(tree):
                self.__makeRules(base, subtree)

    def extract(self, tree, acceptedEDUs):

        self.rules = []
        self.acceptedEDUs = acceptedEDUs
        self.__processTree(tree)

        return self.rules

        # INFO: reguly s� determinowane przez kolejnosc odwiedzania w�z��w przy preprocessingu
        #   Je�li olejemy warto�� relacji, regu�y mog� by� budowane od danego numeru do konca numerk�w:
        #   k -> k+1, k -> k+2, ... k -> n
