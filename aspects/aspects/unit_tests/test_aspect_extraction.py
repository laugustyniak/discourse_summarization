import unittest

from aspects.aspects.aspect_extractor import AspectExtractor
from aspects.configs.conceptnets_config import SENTIC_ASPECTS, \
    CONCEPTNET_ASPECTS, CONCEPTNET_LANG
from aspects.preprocessing.preprocesser import Preprocesser


class AspectExtractionTest(unittest.TestCase):
    def test_get_aspects_nouns(self):
        aspects_expected = [u'car', u'phone']
        preprocesser = Preprocesser()
        raw_text = u'i have car and phone'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        # _ we don not want any concepts to test now
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_noun_phrase(self):
        aspects_expected = [u'car', u'phone']
        preprocesser = Preprocesser()
        raw_text = u'i have a nice car and awesome phone'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_ner(self):
        aspects_expected = [u'angela merkel', u'merkel', u'europe']
        preprocesser = Preprocesser()
        # sample text, don't correct :)
        raw_text = u'Angela Merkel is German, angela merkel is europe!'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_1_char_aspects_and_donts(self):
        aspects_expected = []
        preprocesser = Preprocesser()
        raw_text = u'I don\'t like it, it is not, do not'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_telecom_lower_cased(self):
        aspects_expected = [u'plan', u'sprint']
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better plan and ' \
                   u'encourage me to not leave for sprint i could get a ' \
                   u'better plan there'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_sentic_concept_extraction(self):
        concept = 'screen'
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better screen'
        text = preprocesser.preprocess(raw_text)
        if SENTIC_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            self.assertTrue(
                True if concept in concepts_obtained[
                    'sentic'].keys() else False)
            self.assertEquals(
                len(concepts_obtained['sentic'][concept][concept]), 5)

    def test_conceptnet_io_concept_extraction(self):
        concept = 'screen'
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better screen'
        text = preprocesser.preprocess(raw_text)
        if CONCEPTNET_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            self.assertTrue(
                True if concept in concepts_obtained[
                    'conceptnet_io'].keys() else False)
            # get at least one concept
            self.assertGreater(len(concepts), 0)
            # check if all keys in dictionary are in dict with concepts,
            # keys like in ConceptNet: start, end, relation
            for k in ['start', 'end', 'relation']:
                self.assertTrue(
                    True if k in concepts[0].keys() else False)

    def test_conceptnet_io_concept_extraction_en_filtered(self):
        concept = 'converter'
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better converter'
        text = preprocesser.preprocess(raw_text)
        if CONCEPTNET_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            for c in concepts:
                self.assertEqual(True, CONCEPTNET_LANG == c['start-lang'])
                self.assertEqual(True, CONCEPTNET_LANG == c['end-lang'])

    def test_conceptnet_io_concept_extraction_en_filtered_phone(self):
        concept = 'phone'
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better phone'
        text = preprocesser.preprocess(raw_text)
        if CONCEPTNET_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            for c in concepts:
                self.assertEqual(True, CONCEPTNET_LANG == c['start-lang'])
                self.assertEqual(True, CONCEPTNET_LANG == c['end-lang'])

    def test_conceptnet_io_concept_extraction_paggination(self):
        concept = 'phone'
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better phone'
        text = preprocesser.preprocess(raw_text)
        if CONCEPTNET_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(
                text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            self.assertGreater(len(concepts), 20)

    def test_conceptnet_io_concept_extraction_paggination_same_concepts(self):
        concept = 'device'
        preprocesser = Preprocesser()
        raw_text = u'this device is really good device, but this phone'
        text = preprocesser.preprocess(raw_text)
        if CONCEPTNET_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(
                text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            self.assertGreater(len(concepts), 20)

    def test_keyword_aspect_extraction(self):
        keywords_expected_rake = [(u'propose', 1.0), (u'screen', 1.0)]
        preprocesser = Preprocesser()
        raw_text = u'i wonder if you can propose for me better screen'
        text = preprocesser.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        _, _, keywords_obtained = aspects_extractor.extract(text)
        # check if rake key exists
        self.assertTrue([True if 'rake' in keywords_obtained.keys() else False])
        self.assertEquals(keywords_obtained['rake'], keywords_expected_rake)
