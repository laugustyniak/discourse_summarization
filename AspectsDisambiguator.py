# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import jellyfish

class AspectsDisambiguator(object):

    def __init__(self):
        self.__threshold = 0.75

    def __getJaro(self, word1, word2):
        """
        Calculate Jaro-Winkler distance between two words
        """
        return jellyfish.jaro_winkler(unicode(word1), unicode(word2))

    def __getMostSimilar(self, aspects, aspect):
        """
        Returns aspect most similar to given one
        """
        aspects.remove(aspect)
        similarities = [self.__getJaro(aspect, x) for x in aspects]

        maxIndex = similarities.index(max(similarities))

        return aspects[maxIndex], similarities[maxIndex]

    def process(self, aspects):

        acceptedAspects = []

        for aspect in aspects:
            candidate, similarity = self.__getMostSimilar(aspects, aspect)

            if similarity < self.__threshold and candidate not in acceptedAspects:
                acceptedAspects.append(candidate)
            else:
                acceptedAspects.append(aspect)

        return acceptedAspects
