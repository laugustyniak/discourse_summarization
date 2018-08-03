from hamcrest import assert_that, equal_to

from aspects.preprocessing.conll import Conll
from aspects.preprocessing.transform_formats import TextTag


def test_read_conll_file():
    conll = Conll(file_path='test.conll', n_tag_fields=2)
    conll_sentences = conll.read_file()
    assert_that(len(conll_sentences), equal_to(4))  # number of document


def test_extract_words_and_tags():
    expected_text_tags = [
        TextTag(text='cord', tag='aspect'),
        TextTag(text='battery life', tag='aspect'),
        TextTag(text='tech guy', tag='aspect'),
        TextTag(text='service center', tag='aspect'),
        TextTag(text="`` sales '' team", tag='aspect'),
        TextTag(text="battery", tag='aspect')
    ]

    conll = Conll(file_path='test.conll', n_tag_fields=2)
    conll_sentences = conll.read_file()

    obtained_text_tag = list(conll.extract_words_and_tags(conll_sentences))

    assert_that(obtained_text_tag, equal_to(expected_text_tags))  # number of document
