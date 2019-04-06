import sys

from aspects.preprocessing import preprocessing
from aspects.utilities import settings

sys.path.append(str(settings.EDU_DEPENDENCY_PARSER_PATH))
from trees.parse_tree import ParseTree


class EDUTreePreprocesser:

    def __init__(self):
        self.edus = []

    def process_tree(self, tree, document_id):
        for index, subtree in enumerate(tree):
            if isinstance(subtree, ParseTree):
                self.process_tree(subtree, document_id)
            else:
                subtree = subtree[2:-2]
                extraction_result = preprocessing.preprocess(subtree)
                tree[index] = len(self.edus)
                extraction_result['source_document_id'] = document_id
                self.edus.append(extraction_result)

    def get_preprocessed_edus(self):
        """

        :return: dict [edu_id, edu]
        """
        return {
            idx: edu
            for idx, edu
            in enumerate(self.edus)
        }
