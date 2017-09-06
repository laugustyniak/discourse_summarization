# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda
# update: Lukasz Augustyniak

import logging
import requests

import RAKE

from aspects.configs.conceptnets_config import CONCEPTNET_ASPECTS
from aspects.configs.conceptnets_config import SENTIC_ASPECTS, \
    SENTIC_EXACT_MATCH_CONCEPTS, CONCEPTNET_URL, CONCEPTNET_RELATIONS, \
    CONCEPTNET_LANG, CONCEPTNET_API_URL
from aspects.enrichments.conceptnets import Sentic

log = logging.getLogger(__name__)


class AspectExtractor(object):
    """ Extract aspects from EDU. """

    def __init__(self, ner_types=None, aspects_to_skip=None):
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
                                    u'', u't',
                                    u'noise',
                                    u'customer', u'agent',
                                    u'unk',
                                    u'password',
                                    u'don',
                                    ]
        if ner_types is None:
            self.ner_types = [u'PERSON', u'GPE', u'ORG',
                              u'PRODUCT', u'FAC', u'LOC']
        else:
            self.ner_types = ner_types

        self.Rake = RAKE.Rake(RAKE.SmartStopList())

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

        concepts : dict of dicts
            Dictionary with concept name and dict with concepts and semantically
            related concepts from choosen conceptnet.
            {'sentic':
                {'screen': // concept part name
                    {'screen': ['display', 'pixel', ...]}}

        """
        tokens = input_text['tokens']
        text = input_text['raw_text']
        aspect_sequence = []
        aspect_sequence_main_encountered = False
        aspect_sequence_enabled = False
        concept_aspects = {}

        # todo add config with NER flag
        # 1. look for NER examples
        # aspects = [ent.text for ent in tokens.ents
        # if ent.label_ in self.ner_types]
        aspects = input_text['entities']

        # 2. NOUN and NOUN phrases
        for idx, token in enumerate(tokens):
            # jesli jest główny (rzeczownik) - akceptujemy od razu
            if self._is_interesting_main(token) and len(token) > 1:
                if not token['is_stop']:
                    aspect_sequence.append(token['lemma'])
                aspect_sequence_enabled = True
                aspect_sequence_main_encountered = True
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

        # lower case every aspect and only longer than 1
        aspects = [x.strip().lower() for x in aspects
                   if x not in self.aspects_to_skip and x != '']

        # 3. senticnet
        if SENTIC_ASPECTS:
            sentic = Sentic()
            # dict with concepts and related concepts
            concept_aspects_ = {}
            for asp in aspects:
                asp = asp.replace(' ', '_')
                concept_aspects_[asp] = \
                    sentic.get_semantic_concept_by_concept(asp,
                                                           SENTIC_EXACT_MATCH_CONCEPTS)
            # save conceptnet name
            concept_aspects['sentic'] = concept_aspects_

        # 4. ConceptNet.io
        if CONCEPTNET_ASPECTS:
            concept_aspects_ = {}
            for asp in aspects:
                concept_aspects_[asp] = []
                next_page = CONCEPTNET_URL + asp + u'?offset=0&limit=20'
                while next_page:
                    response = requests.get(next_page).json()
                    cn_view = response['view']
                    cn_edges = response['edges']
                    try:
                        next_page = CONCEPTNET_API_URL + cn_view['nextPage']
                        log.info(
                            'Next page from ConceptNet.io: {}'.format(
                                next_page))
                    except KeyError:
                        next_page = None

                    for edge in cn_edges:
                        relation = edge['rel']['label']
                        if relation in CONCEPTNET_RELATIONS \
                                and (edge['start'][
                                         'language'] == CONCEPTNET_LANG
                                     and edge['end'][
                                        'language'] == CONCEPTNET_LANG):
                            concept_aspects_[asp].append(
                                {'start': edge['start']['label'].lower(),
                                 'start-lang': edge['start']['language'],
                                 'end': edge['end']['label'].lower(),
                                 'end-lang': edge['end']['language'],
                                 'relation': relation,
                                 'weight': edge['weight']})
            concept_aspects['conceptnet_io'] = concept_aspects_

        # 5. keyword extraction
        if text:
            keyword_aspects = {'rake': self.Rake.run(text)}
        else:
            keyword_aspects = {'rake': [(None, None)]}

        return aspects, concept_aspects, keyword_aspects
