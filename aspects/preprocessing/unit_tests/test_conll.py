from hamcrest import assert_that

from aspects.preprocessing.conll import Conll


def test_read_conll_file():
    conll = Conll(file_path='test.conll', n_tag_fields=2).read_file()
    assert_that(len(conll), 4)  # number of document
