import unittest

from parse import DiscourseParser
from trees.parse_tree import ParseTree


class RSTParseTest(unittest.TestCase):
    def _get_document(self):
        self.document = """
        If you can get past this little flaw, give the ipod 5 stars, because everything else about it is pretty  
        much perfect.
        """

    def _get_simple_sentences_document(self):
        self.document = """
        
        """

    def test_parse_document(self):
        parser = DiscourseParser(output_dir='', global_features=True)
        parsed_tree = parser.parse('0')
        self.assertTrue(isinstance(parsed_tree, ParseTree))
