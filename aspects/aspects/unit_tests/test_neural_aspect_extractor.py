from aspects.aspects.neural_aspect_extractor import NeuralAspectExtractor
from hamcrest import assert_that, equal_to


def test_aspect_extraction():
    nae = NeuralAspectExtractor()
    aspects_obtained = nae.extract('has a really good screen quality')
    assert_that(aspects_obtained, equal_to(['screen quality']))
