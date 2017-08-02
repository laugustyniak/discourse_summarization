# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda
# update: Lukasz Augustyniak

from aspects.conceptnets import Sentic


class AspectExtractor(object):
    """ Extract aspects from EDU. """

    def __init__(self, ner_types=None, aspects_to_skip=None,
                 conceptnet_io=False, senticnet=False):
        """
        Initialize extractor aspect extractor.

        Parameters
        ----------
        ner_types : list
            List of unicode names of Named Entity than may by be chosen
            by spacy extractor as aspects. See
            https://spacy.io/docs/usage/entity-recognition#entity-types
            for more information

        aspects_to_skip : list
            List of aspects that should be removed.

        conceptnet_io : bool
            Do we use ConceptNet based entities in aspect extraction procedure.
            By default False.
            We use conceptnet.io as data source here.

        senticnet : bool
            Do we use sentic conceptnet based entities in aspect extraction
            procedure. By default False. We use sentic.net as data source here.
        """
        if aspects_to_skip is not None:
            self.aspects_to_skip = aspects_to_skip
        else:
            self.aspects_to_skip = [u'day', u'days', u'week', u'weeks',
                                    u'tonight',
                                    u'total', u'laughter', u'tongue',
                                    u'weekend', u'month', u'months', u'year',
                                    u'years', u'time', u'today', u'data',
                                    u'date',
                                    u'monday', u'tuesday', u'wednesday',
                                    u'thursday', u'friday', u'saturday',
                                    u'sunday',
                                    u'january', u'february', u'march', u'april',
                                    u'may', u'june', u'july', u'august',
                                    u'september', u'october',
                                    u'november', u'december',
                                    u'end',
                                    u'',
                                    u'noise',
                                    u'customer', u'agent',
                                    u'unk',
                                    u'password',
                                    ]
        if ner_types is None:
            self.ner_types = [u'PERSON', u'GPE', u'ORG',
                              u'PRODUCT', u'FAC', u'LOC']
        else:
            self.ner_types = ner_types
        self.conceptnet_io = conceptnet_io
        self.senticnet = senticnet

    def _is_interesting_main(self, token):
        return token['pos'] == 'NOUN'

    def _is_interesting_addition(self, token):
        return token['pos'] == 'ADV' \
               or token['pos'] == 'NUM' \
               or token['pos'] == 'NOUN' \
               or token['pos'] == 'ADJ'

    def extract(self, input_text):
        """
        Extracts all possible aspects - NER, NOUN and NOUN PHRASES,
        potentially other dictionary based aspects.

        Parameters
        ----------
        input_text : dictionary
            Dictionary with raw text and spacy object with each
            token information.

        Returns
        ----------
        aspects : list
            List of extracted aspects.

        """
        tokens = input_text['tokens']
        aspect_sequence = []
        aspect_sequence_main_encountered = False
        aspect_sequence_enabled = False
        concept_aspects = {}

        # 1. look for NER examples
        # aspects = [ent.text for ent in tokens.ents
        # if ent.label_ in self.ner_types]
        aspects = input_text['entities']

        # 2. NOUN and NOUN phrases
        for idx, token in enumerate(tokens):
            # jesli jest główny (rzeczownik) - akceptujemy od razu
            if self._is_interesting_main(token):
                if not token['is_stop']:
                    aspect_sequence.append(token['lemma'])
                aspect_sequence_enabled = True
                aspect_sequence_main_encountered = True

            # jesli jest ciekawy (przymiotnik, przysłówek, liczba)
            # i jest potencjalnym elementem sekwencji - dodajemy
            elif self._is_interesting_addition(token) and (
                        (idx + 1 < len(tokens)
                         and self._is_interesting_addition(tokens[idx + 1]))
                    or (idx + 1 == len(tokens))):
                if not token['is_stop']:
                    aspect_sequence.append(token['lemma'])
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

        # 3. senticnet
        if self.senticnet:
            # todo: how deal with several concepts for specific aspect?
            sentic = Sentic()
            concept_aspects = {}
            for asp in aspects:
                asp = asp.replace(' ', '_')
                concept_aspects[asp] = \
                    sentic.get_semantic_concept_by_concept(asp)

        # 4. ConceptNet.io
        if self.conceptnet_io:
            # todo: impl
            # lista aspektów
            pass

        # nie wiem czemu puste wartosci leca - odfiltrowujemy
        # lower case every aspect and only longer than 1
        aspects = [x.strip().lower() for x in aspects
                   if x not in self.aspects_to_skip]

        return aspects, concept_aspects
