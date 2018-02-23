import unittest

from parse import DiscourseParser
from trees.parse_tree import ParseTree


class RSTParseTest(unittest.TestCase):

    def _with_one_sentence_document(self):
        self.document = u"""
        If you can get past this little flaw, give the ipod 5 stars, because everything else about it is pretty  
        much perfect.
        """

    def _with_simple_sentences_document(self):
        self.document = u"""crappy battery. crappy screen. i love apple."""

    def _with_brexit_document(self):
        self.document = u"""He made his argument during a debate on the proposed transition period after Britain 
        leaves the EU next year. Labour MPs have repeatedly moaned that the Government needs to outline its plans for 
        the EU negotiation as there is now just over a year until the UK officially withdraws from the trade bloc.
        However, countering the narrative, one member of the audience said: There's this preoccupation that time is 
        running out for us. But time's running out for French farmers, Germany car manufactures, for Spanish tourist 
        resorts, and last of all, time's running out for Italian politicians."""

    def test_parse_document_str_one_sentence(self):
        parser = DiscourseParser(output_dir='', global_features=True)
        self._with_one_sentence_document()
        parsed_tree = parser.parse(text=self.document)
        self.assertTrue(isinstance(parsed_tree, ParseTree))

    def test_parse_document_str_multi_sentence(self):
        parser = DiscourseParser(output_dir='', global_features=True)
        self._with_brexit_document()
        parsed_tree = parser.parse(text=self.document)
        self.assertTrue(isinstance(parsed_tree, ParseTree))
