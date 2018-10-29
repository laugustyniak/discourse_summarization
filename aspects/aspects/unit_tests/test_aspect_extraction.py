import pytest
from hamcrest import assert_that, equal_to, greater_than, is_

from aspects.aspects.aspect_extractor import AspectExtractor
from aspects.utilities import settings


@pytest.mark.parametrize('aspects_expected, text', [
    (
            [u'car', u'phone'],
            u'i have car and phone'
    ),
    (
            [u'car', u'phone'],
            u'i have a nice car and awesome phone'
    ),
    (
            [u'angela merkel', u'europe', u'merkel'],
            u'Angela Merkel is German, merkel is europe!'
    ),
    (
            [u'phone', u'sound', u'bass'],
            u'this phone has excellent sound and bass'
    ),
    (
            [u'iphone'],
            u'i wonder if you can propose iphone x'
    ),
    (
            [u'iphone'],
            u'i wonder if you can propose blue iphone'
    ),
    (
            [u'word explorer netscape'],
            u'i wonder if you can propose word explorer netscape acrobat reader photoshop'
    ),
    (
            [u'plan', u'sprint'],
            u'i wonder if you can propose for me better plan and encourage me to not leave for sprint i could get '
            u'a better plan there'
    ),
    (
            [u'boxwave bodysuit premium', u'tpu rubber gel'],
            u'this is boxwave bodysuit premium textured tpu rubber gel'
    )
])
def test_get_aspects_nouns(aspects_expected, text):
    aspects_extractor = AspectExtractor(sentic=False, conceptnet=False)
    # _ we don not want any concepts to test now
    aspects_obtained, _, _ = aspects_extractor.extract(text)
    assert_that(set(aspects_obtained), equal_to(set(aspects_expected)))


@pytest.mark.parametrize('text', [
    u'I don\'t like it, it is not, do not',
    u'thank you',
    u'this is t |',
])
def test_not_aspect_extracted(text):
    aspects_extractor = AspectExtractor(sentic=False, conceptnet=False)
    # _ we don not want any concepts to test now
    aspects_obtained, _, _ = aspects_extractor.extract(text)
    assert_that(aspects_obtained, equal_to([]))


def test_get_aspects_ner_false():
    text = u'this is the biggest in Europe!'
    aspects_extractor = AspectExtractor(is_ner=False, sentic=False, conceptnet=False)
    assert_that(aspects_extractor.extract(text)[0], equal_to([]))


def test_sentic_concept_extraction():
    concept = u'screen'
    text = u'i wonder if you can propose for me better screen'
    if settings.SENTIC_ASPECTS:
        aspects_extractor = AspectExtractor(conceptnet=False)
        _, concepts_obtained, _ = aspects_extractor.extract(text)
        is_(True if concept in concepts_obtained['sentic'].keys() else False)
        assert_that(len(concepts_obtained['sentic'][concept][concept]), equal_to(5))


def test_conceptnet_io_concept_extraction():
    concept = 'screen'
    text = u'i wonder if you can propose for me better screen'
    if settings.CONCEPTNET_IO_ASPECTS:
        aspects_extractor = AspectExtractor(sentic=False)
        _, concepts_obtained, _ = aspects_extractor.extract(text)
        concepts = concepts_obtained['conceptnet_io'][concept]

        greater_than(len(concepts), 0)
        # check if all keys in dictionary are in dict with concepts,
        # keys like in ConceptNet: start, end, relation
        for k in ['start', 'end', 'relation']:
            is_(True if k in concepts[0].keys() else False)


def test_conceptnet_io_concept_extraction_en_filtered():
    concept = 'converter'
    text = u'i wonder if you can propose for me better converter'
    if settings.CONCEPTNET_IO_ASPECTS:
        aspects_extractor = AspectExtractor(sentic=False)
        _, concepts_obtained, _ = aspects_extractor.extract(text)
        concepts = concepts_obtained['conceptnet_io'][concept]

        for c in concepts:
            assert_that(True, equal_to(settings.CONCEPTNET_IO_LANG == c['start-lang']))
            assert_that(True, equal_to(settings.CONCEPTNET_IO_LANG == c['end-lang']))


def test_conceptnet_io_concept_extraction_en_filtered_phone():
    concept = 'phone'
    text = u'i wonder if you can propose for me better phone'
    if settings.CONCEPTNET_IO_ASPECTS:
        aspects_extractor = AspectExtractor(sentic=False)
        _, concepts_obtained, _ = aspects_extractor.extract(text)
        concepts = concepts_obtained['conceptnet_io'][concept]

        for c in concepts:
            assert_that(True, settings.CONCEPTNET_IO_LANG == c['start-lang'])
            assert_that(True, settings.CONCEPTNET_IO_LANG == c['end-lang'])


def test_keyword_aspect_extraction():
    keywords_expected_rake = [(u'propose', 1.0), (u'screen', 1.0)]
    text = u'i wonder if you can propose for me better screen'
    aspects_extractor = AspectExtractor()
    _, _, keywords_obtained = aspects_extractor.extract(text)
    # check if rake key exists
    is_([True if 'rake' in keywords_obtained.keys() else False])
    assert_that(keywords_obtained['rake'], equal_to(keywords_expected_rake))
