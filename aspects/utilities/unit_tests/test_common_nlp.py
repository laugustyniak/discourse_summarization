from aspects.utilities.common_nlp import spelling


def test_spelling():
    word = 'ipod'
    corrected_word = 'iPod'
    word_obtained = spelling(word)
    assert corrected_word == word_obtained
