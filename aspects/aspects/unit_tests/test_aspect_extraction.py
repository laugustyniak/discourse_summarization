import unittest

from aspects.aspects.aspect_extractor import AspectExtractor
from aspects.configs.conceptnets_config import SENTIC_ASPECTS
from aspects.preprocessing.preprocesser import Preprocesser


class AspectExtractionTest(unittest.TestCase):
    def test_get_aspects_nouns(self):
        aspects_expected = [u'car', u'phone']
        preprocesser = Preprocesser()
        raw_text = u'i have car and phone'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        # _ we don not want any concepts to test now
        aspects_obtained, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_noun_phrase(self):
        aspects_expected = [u'car', u'phone']
        preprocesser = Preprocesser()
        raw_text = u'i have a nice car and awesome phone'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_ner(self):
        aspects_expected = [u'angela merkel', u'merkel', u'europe']
        preprocesser = Preprocesser()
        # sample text, don't correct :)
        raw_text = u'Angela Merkel is German, angela merkel is europe!'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_telecom_lower_cased(self):
        aspects_expected = [u'plan', u'sprint']
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better plan and ' \
                   u'encourage me to not leave for sprint i could get a ' \
                   u'better plan there'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_sentic_concept_extraction(self):
        concept = 'screen'
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better screen'
        text = preprocesser.preprocess(raw_text)
        if SENTIC_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained = aspects_extractor.extract(text)
            self.assertTrue(
                True if concept in concepts_obtained[
                    'sentic'].keys() else False)
            self.assertEquals(
                len(concepts_obtained['sentic'][concept][concept]), 5)
