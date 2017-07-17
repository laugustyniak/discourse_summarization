# -*- coding: utf-8 -*-

from os.path import basename, join, exists
from os import system, makedirs
from glob import glob
from pprint import pprint

from IPython.display import Image, display
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from nltk.tree import Tree

import sys
import re

from wordcloud import WordCloud
from bs4 import BeautifulSoup

import requests
import simplejson
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append("../edu_dependency_parser/src")

from trees.parse_tree import ParseTree

from utils import load_serialized, flatten_list

data_path = '../results/ipod'
# data_path = '../results/Canon_S100'
dataset_name = basename(data_path)
data_path_trees = join(data_path, 'edu_trees_dir')
data_path_link_tree = join(data_path, 'link_trees_dir')

tree_names = [basename(x) for x in glob(join(data_path_trees, '*.tree.ser'))]
n_docs = len(tree_names)

documents_info = load_serialized(join(data_path, 'documents_info'))
document_info_at_least_2_aspects_accepted = {k: v for k, v in documents_info.items() if len(v['accepted_edus']) > 1}
n_at_least_two_aspects_per_doc = len(document_info_at_least_2_aspects_accepted)


class EDUTreeRulesExtractor(object):
    def __init__(self):
        self.rules = []
        self.accepted_edus = None
        self.relation = None

    def __process_tree(self, tree, relation):
        for index, node in enumerate(tree):
            # go to the subtree
            # process tree down
            if isinstance(node, ParseTree):
                self.__process_tree(node, relation)
            #                 print '++++', tree.node, node
            # recursively until leaf
            else:
                # leaf, parent/current subtree, index of leaf in the tree
                self.__traverse_parent(node, tree, index, tree.node)
            #                 print '****', node, tree, index, tree.node

    def __traverse_parent(self, leaf, parent, child_index, relation):
        # leaf = child
        if parent is not None:
            for id in range(child_index + 1, len(parent)):
                self.__make_rules(leaf, parent[id], relation)

            if parent.parent is not None:
                self.__traverse_parent(leaf, parent.parent, parent.parent_index, relation)
            else:
                self.relation = parent.node
            #                 print 'the highest common node', relation, leaf, parent

    def __make_rules(self, leaf_left, tree, relation):

        # int oznacza, że rekursywnie doszliśmy do liścia w drzewie pod aktualnym liście analizowanym
        if isinstance(tree, int):
            if tree in self.accepted_edus and leaf_left in self.accepted_edus:
                print('leaf leaft', leaf_left, 'leaf right', tree, 'relation', relation)
                self.rules.append((leaf_left, tree, relation))
        else:
            for index, subtree in enumerate(tree):
                self.__make_rules(leaf_left, subtree, tree.node)
                print '--', tree.node, leaf_left, subtree

    def extract(self, tree, accepted_edus):
        self.accepted_edus = accepted_edus
        print 'root node', tree.node
        self.__process_tree(tree, tree.node)

        return self.rules

        # INFO: reguly są determinowane przez kolejnosc odwiedzania węzłów przy preprocessingu
        #   Jeśli olejemy wartość relacji, reguły mogą być budowane od danego numeru do konca numerków:
        #   k -> k+1, k -> k+2, ... k -> n

for doc_number in document_info_at_least_2_aspects_accepted.keys()[11:]:
    rules_extractor = EDUTreeRulesExtractor()
    document_info = load_serialized(join(data_path, 'documents_info'))[doc_number]
#     print(document_info)
    link_tree = load_serialized(join(data_path_link_tree, str(doc_number)))
#     print(link_tree)
    print('====='*20)
    extracted_rules = rules_extractor.extract(link_tree, document_info['accepted_edus'])
    break