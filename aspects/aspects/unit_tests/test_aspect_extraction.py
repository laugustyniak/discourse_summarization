import unittest

from aspects.aspects.aspect_extractor import AspectExtractor
from aspects.preprocessing import preprocessing
from aspects.utilities import settings


class AspectExtractionTest(unittest.TestCase):

    def test_get_aspects_nouns(self):
        aspects_expected = [u'car', u'phone']
        raw_text = u'i have car and phone'
        text = preprocessing.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        # _ we don not want any concepts to test now
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_noun_phrase(self):
        aspects_expected = [u'car', u'phone']
        raw_text = u'i have a nice car and awesome phone'
        text = preprocessing.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_ner(self):
        aspects_expected = [u'angela merkel', u'europe']
        # sample text, don't correct :)
        raw_text = u'Angela Merkel is German, angela merkel is europe!'
        text = preprocessing.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(set(aspects_obtained), set(aspects_expected))

    def test_get_aspects_ner_false(self):
        aspects_expected = []
        # sample text, don't correct :)
        raw_text = u'this is the biggest in Europe!'
        text = preprocessing.preprocess(raw_text)
        aspects_extractor = AspectExtractor(is_ner=False)
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_1_char_aspects_and_donts(self):
        aspects_expected = []
        raw_text = u'I don\'t like it, it is not, do not'
        text = preprocessing.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_get_aspects_telecom_lower_cased(self):
        aspects_expected = [u'plan', u'sprint']
        raw_text = u'i wonder if you can propose for me better plan and ' \
                   u'encourage me to not leave for sprint i could get a ' \
                   u'better plan there'
        text = preprocessing.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        aspects_obtained, _, _ = aspects_extractor.extract(text)
        self.assertEqual(set(aspects_obtained), set(aspects_expected))

    def test_extract_noun_and_noun_phrases_very_long_aspects_trimming(self):
        aspects_expected = [u'word explorer netscape']
        text = u'i wonder if you can propose word explorer netscape acrobat reader photoshop'
        aspects_extractor = AspectExtractor()
        aspects_obtained = aspects_extractor.extract_noun_and_noun_phrases(text)
        self.assertEqual(aspects_expected, aspects_obtained)

    def test_extract_noun(self):
        aspects_expected = [u'iphone']
        text = u'i wonder if you can propose blue iphone'
        aspects_extractor = AspectExtractor()
        aspects_obtained = aspects_extractor.extract_noun_and_noun_phrases(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_extract_noun_phrases_with_number(self):
        aspects_expected = [u'iphone x']
        text = u'i wonder if you can propose iphone X'
        aspects_extractor = AspectExtractor()
        aspects_obtained = aspects_extractor.extract_noun_and_noun_phrases(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_extract_noun_phrases_skip_ps(self):
        aspects_expected = []
        text = u'I propose p.s. <'
        aspects_extractor = AspectExtractor()
        aspects_obtained = aspects_extractor.extract_noun_and_noun_phrases(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_extract_nouns(self):
        aspects_expected = [u'phone', u'sound', u'bass']
        text = u'this phone has excellent sound and bass'
        aspects_extractor = AspectExtractor()
        aspects_obtained = aspects_extractor.extract_noun_and_noun_phrases(text)
        self.assertEqual(aspects_obtained, aspects_expected)

    def test_sentic_concept_extraction(self):
        concept = 'screen'
        raw_text = u'i wonder if you can propose for me better screen'
        text = preprocessing.preprocess(raw_text)
        if settings.SENTIC_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            self.assertTrue(True if concept in concepts_obtained['sentic'].keys() else False)
            self.assertEquals(len(concepts_obtained['sentic'][concept][concept]), 5)

    def test_conceptnet_io_concept_extraction(self):
        concept = 'screen'
        raw_text = u'i wonder if you can propose for me better screen'
        text = preprocessing.preprocess(raw_text)
        if settings.CONCEPTNET_IO_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            self.assertGreater(len(concepts), 0)
            # check if all keys in dictionary are in dict with concepts,
            # keys like in ConceptNet: start, end, relation
            for k in ['start', 'end', 'relation']:
                self.assertTrue(True if k in concepts[0].keys() else False)

    def test_conceptnet_io_concept_extraction_en_filtered(self):
        concept = 'converter'
        raw_text = u'i wonder if you can propose for me better converter'
        text = preprocessing.preprocess(raw_text)
        if settings.CONCEPTNET_IO_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            for c in concepts:
                self.assertEqual(True, settings.CONCEPTNET_IO_LANG == c['start-lang'])
                self.assertEqual(True, settings.CONCEPTNET_IO_LANG == c['end-lang'])

    def test_conceptnet_io_concept_extraction_en_filtered_phone(self):
        concept = 'phone'
        raw_text = u'i wonder if you can propose for me better phone'
        text = preprocessing.preprocess(raw_text)
        if settings.CONCEPTNET_IO_ASPECTS:
            aspects_extractor = AspectExtractor()
            _, concepts_obtained, _ = aspects_extractor.extract(text)
            concepts = concepts_obtained['conceptnet_io'][concept]

            for c in concepts:
                self.assertEqual(True, settings.CONCEPTNET_IO_LANG == c['start-lang'])
                self.assertEqual(True, settings.CONCEPTNET_IO_LANG == c['end-lang'])

    def test_keyword_aspect_extraction(self):
        keywords_expected_rake = [(u'propose', 1.0), (u'screen', 1.0)]
        raw_text = u'i wonder if you can propose for me better screen'
        text = preprocessing.preprocess(raw_text)
        aspects_extractor = AspectExtractor()
        _, _, keywords_obtained = aspects_extractor.extract(text)
        # check if rake key exists
        self.assertTrue([True if 'rake' in keywords_obtained.keys() else False])
        self.assertEquals(keywords_obtained['rake'], keywords_expected_rake)
