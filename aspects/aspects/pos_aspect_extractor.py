# -*- coding: utf-8 -*-


class POSAspectExtractor(object):
    def _is_interesting_main(self, token):
        return token['pos'] == 'NOUN'

    def _is_interesting_addition(self, token):
        return token['pos'] == 'ADV' or token['pos'] == 'NUM' \
               or token['pos'] == 'NOUN' or token['pos'] == 'ADJ'

    def extract(self, input_data):

        tokens = input_data['tokens']

        aspects = []
        aspect_sequence = []
        aspect_sequence_main_encountered = False
        aspect_sequence_enabled = False

        # dla każdego tokena tekstu
        for idx, result in enumerate(tokens):

            # jesli jest główny (rzeczownik) - akceptujemy od razu
            if self._is_interesting_main(result):
                if not result['is_stop']:
                    aspect_sequence.append(result['text'])
                aspect_sequence_enabled = True
                aspect_sequence_main_encountered = True

            # jesli jest ciekawy (przymiotnik, przysłówek, liczba) i jest
            # potencjalnym elementem sekwencji - dodajemy
            elif self._is_interesting_addition(result) and (
                        (idx + 1 < len(
                            tokens) and self._is_interesting_addition(
                            tokens[idx + 1])) or idx + 1 == len(
                        tokens)):
                if not result['is_stop']:
                    aspect_sequence.append(result['text'])
            else:

                # akceptujemy sekwencje, jesli byl w niej element główny
                if aspect_sequence_enabled and aspect_sequence_main_encountered:
                    aspect = ' '.join(aspect_sequence)

                    if aspect not in aspects:
                        aspects.append(aspect)

                aspect_sequence_main_encountered = False
                aspect_sequence_enabled = False
                aspect_sequence = []

        # dodajemy ostatnią sekwencje
        if aspect_sequence_enabled and aspect_sequence_main_encountered:
            aspects.append(' '.join(aspect_sequence))

        # nie wiem czemu puste wartosci leca - odfiltrowujemy
        aspects = [x for x in aspects if len(x) > 0]

        return aspects
