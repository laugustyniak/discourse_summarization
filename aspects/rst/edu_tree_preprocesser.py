import os
import sys

from aspects.preprocessing.preprocesser import preprocess

sys.path.append(os.getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree


def process_tree(tree):
    for index, subtree in enumerate(tree):
        if isinstance(subtree, ParseTree):
            process_tree(subtree)
        else:
            preprocessed_edu = preprocess(subtree)
            yield preprocessed_edu
