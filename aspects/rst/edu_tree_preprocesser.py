import os
import sys
import uuid

from aspects.preprocessing.preprocesser import preprocess

sys.path.append(os.getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree


class EduExtractor:

    def __init__(self):
        self.edus = []

    def process_tree(self, tree):
        for index, subtree in enumerate(tree):
            if isinstance(subtree, ParseTree):
                self.process_tree(subtree)
            else:
                preprocessed_edu = preprocess(subtree)
                self.edus.append(preprocessed_edu)

    def get_edus_with_idsfrom_tree(self, tree):
        self.edus = []
        self.process_tree(tree)
        return {uuid.uuid4(): edu for edu in self.edus}
