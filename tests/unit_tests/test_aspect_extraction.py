import unittest

from AspectExtractor import AspectExtractor
from Preprocesser import Preprocesser


class AspectExtractionTest(unittest.TestCase):

    def test_get_aspects_nouns(self):
        aspects_expected = [u'car', u'phone']
        preprocesser = Preprocesser()
        raw_text = u'i have car and phone'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained = aspects_extractor.extract(text)
        print aspects_obtained
        print aspects_expected
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_ner(self):
        aspects_expected = [u'angela merkel', u'merkel', u'europe']
        preprocesser = Preprocesser()
        # sample text, don't correct :)
        raw_text = u'Angela Merkel is German, angela merkel is europe!'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained = aspects_extractor.extract(text)
        print aspects_obtained
        print aspects_expected
        self.assertEqual(aspects_obtained, aspects_expected)
