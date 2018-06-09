import pytest

from aspects.utilities.common_nlp import spelling


@pytest.mark.parametrize("word, corrected_word", [
    ('ipod', 'iPod'),
    ('bargins', 'bargain'),
])
def test_spelling(word, corrected_word):
    word_obtained = spelling(word)
    assert corrected_word == word_obtained
