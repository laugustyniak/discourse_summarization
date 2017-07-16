# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from Preprocesser import Preprocesser

import sys
import os

sys.path.append(os.getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree


class EDUTreePreprocesser(object):
    def __init__(self):
        self.edus = []
        self.preprocesser = Preprocesser()

    def processTree(self, tree, document_id):
        for index, subtree in enumerate(tree):
            if isinstance(subtree, ParseTree):
                self.processTree(subtree, document_id)
            else:
                subtree = subtree[2:-2]

                extractionResult = self.preprocesser.preprocess(subtree)

                tree[index] = len(self.edus)

                extractionResult['source_document_id'] = document_id

                self.edus.append(extractionResult)

    def getPreprocessedEdusList(self):

        eduDict = {}

        for id, edu in enumerate(self.edus):
            eduDict[id] = edu

        return eduDict
