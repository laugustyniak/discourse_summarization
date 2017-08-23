# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import jellyfish


class AspectsDisambiguator(object):
    def __init__(self):
        self.__threshold = 0.75

    def _get_jaro(self, word1, word2):
        """
        Calculate Jaro-Winkler distance between two words
        """
        return jellyfish.jaro_winkler(unicode(word1), unicode(word2))

    def _get_most_similar(self, aspects, aspect):
        """
        Returns aspect most similar to given one
        """
        aspects.remove(aspect)
        similarities = [self._get_jaro(aspect, x) for x in aspects]

        max_index = similarities.index(max(similarities))

        return aspects[max_index], similarities[max_index]

    def process(self, aspects):

        accepted_aspects = []

        for aspect in aspects:
            candidate, similarity = self._get_most_similar(aspects, aspect)

            if similarity < self.__threshold \
                    and candidate not in accepted_aspects:
                accepted_aspects.append(candidate)
            else:
                accepted_aspects.append(aspect)

        return accepted_aspects
