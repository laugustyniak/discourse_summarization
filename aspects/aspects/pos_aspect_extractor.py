# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import spacy


class POSAspectExtractor:
    def __isInterestingMain(self, token):
        return token['pos'] == 'NOUN'

    def __isInterestingAddition(self, token):
        return token['pos'] == 'ADV' or token['pos'] == 'NUM' or token['pos'] == 'NOUN' or token['pos'] == 'ADJ'

    def extract(self, inputData):

        tokens = inputData['tokens']

        aspects = []
        aspectSequence = []
        aspectSequenceMainEncountered = False
        aspectSequenceEnabled = False

        # dla każdego tokena tekstu
        for id, result in enumerate(tokens):

            # jesli jest główny (rzeczownik) - akceptujemy od razu
            if self.__isInterestingMain(result):
                if not result['is_stop']:
                    aspectSequence.append(result['text'])
                aspectSequenceEnabled = True
                aspectSequenceMainEncountered = True

            # jesli jest ciekawy (przymiotnik, przysłówek, liczba) i jest potencjalnym elementem sekwencji - dodajemy
            elif self.__isInterestingAddition(result) and (
                        (id + 1 < len(tokens) and self.__isInterestingAddition(tokens[id + 1])) or id + 1 == len(
                        tokens)):
                if not result['is_stop']:
                    aspectSequence.append(result['text'])
            else:

                # akceptujemy sekwencje, jesli byl w niej element główny
                if aspectSequenceEnabled and aspectSequenceMainEncountered:
                    aspect = ' '.join(aspectSequence)

                    if not aspect in aspects:
                        aspects.append(aspect)

                aspectSequenceMainEncountered = False
                aspectSequenceEnabled = False
                aspectSequence = []

        # dodajemy ostatnią sekwencje
        if aspectSequenceEnabled and aspectSequenceMainEncountered:
            aspects.append(' '.join(aspectSequence))

        # nie wiem czemu puste wartosci leca - odfiltrowujemy
        aspects = [x for x in aspects if len(x) > 0]

        return aspects
