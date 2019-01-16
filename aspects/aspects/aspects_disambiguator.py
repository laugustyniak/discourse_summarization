import jellyfish

from aspects.utilities import common_nlp


# TODO: do word embeddings based clustering of close aspects + jaro wrinkler
# TODO: add spelling correction here

def _get_jaro(word1, word2):
    """ Calculate Jaro-Winkler distance between two words """
    return jellyfish.jaro_winkler(unicode(word1), unicode(word2))


def _get_most_similar(aspects, aspect):
    """ Returns aspect most similar to given one """
    aspects.remove(aspect)
    similarities = [_get_jaro(aspect, x) for x in aspects]

    max_index = similarities.index(max(similarities))

    return aspects[max_index], similarities[max_index]


def process(aspects, threshold):
    # TODO: rename function, filter only noun phrases based aspects
    aspects = [common_nlp.spelling(aspect) for aspect in aspects]
    accepted_aspects = []

    for aspect in aspects:
        candidate, similarity = _get_most_similar(aspects, aspect)

        if similarity < threshold and candidate not in accepted_aspects:
            accepted_aspects.append(candidate)
        else:
            accepted_aspects.append(aspect)

    return accepted_aspects
