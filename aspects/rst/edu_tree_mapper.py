import uuid

from nltk import Tree


class EDUTreeMapper:

    def __init__(self):
        self.edus = []

    def process_tree(self, tree):
        for index, subtree in enumerate(tree):
            if isinstance(subtree, Tree):
                self.process_tree(subtree)
            else:
                tree[index] = int(uuid.uuid4())
                self.edus.append(subtree)
