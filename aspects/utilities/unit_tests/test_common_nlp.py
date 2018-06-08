import pytest
from hamcrest import assert_that, equal_to

from aspects.utilities.common_nlp import spelling


@pytest.mark.parametrize("word, corrected_word", [
    ('ipod', 'iPod'),
    ('bargins', 'bargain'),
])
def test_spelling(word, corrected_word):
    word_obtained = spelling(word)
    assert_that(corrected_word, equal_to(word_obtained))
