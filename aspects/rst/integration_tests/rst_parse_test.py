import unittest

import sys
from os import getcwd
sys.path.append(getcwd() + "/edu_dependency_parser/src")
from trees.parse_tree import ParseTree
from parse import DiscourseParser


class RSTParseTest(unittest.TestCase):
    def _get_document(self):
        self.document = """
        If you can get past this little flaw, give the ipod 5 stars, because everything else about it is pretty  
        much perfect.
        """

    def _get_simple_sentences_document(self):
        self.document = """
        
        """

    def test_parse_docuemnt(self):
        parser = DiscourseParser(output_dir=edu_trees_dir,
                                 # verbose=True,
                                 # skip_parsing=True,
                                 # global_features=True,
                                 # save_preprocessed_doc=True,
                                 # preprocesser=None
                                 )