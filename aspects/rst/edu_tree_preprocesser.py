from nltk import Tree

from aspects.preprocessing import preprocessing


class EDUTreePreprocesser:

    def __init__(self):
        self.edus = []

    def process_tree(self, tree):
        for index, subtree in enumerate(tree):
            if isinstance(subtree, Tree):
                self.process_tree(subtree)
            else:
                extraction_result = preprocessing.preprocess(subtree)
                tree[index] = len(self.edus)
                self.edus.append(extraction_result)

    def get_preprocessed_edus(self):
        return {
            idx: edu
            for idx, edu
            in enumerate(self.edus)
        }
