import logging
import sys
from collections import defaultdict

import RAKE

from aspects.enrichments.conceptnets import Sentic, load_conceptnet_io
from aspects.utilities import settings

reload(sys)
sys.setdefaultencoding('utf-8')

log = logging.getLogger(__name__)


class AspectExtractor(object):
    """ Extract aspects from EDU. """

    def __init__(self, ner_types=None, aspects_to_skip=None, is_ner=True):
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

        is_ner : bool
            Do we want to extract Named Entity as aspects?
        """
        self.is_ner = is_ner
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

        if settings.CONCEPTNET_IO_ASPECTS:
            self.conceptnet_io = load_conceptnet_io()

    def _is_interesting_main(self, token):
        return token['pos'] == 'NOUN'

    def _is_interesting_addition(self, token):
        return token['pos'] == 'ADV' \
               or token['pos'] == 'NUM' \
               or token['pos'] == 'NOUN' \
               or token['pos'] == 'ADJ'

    def extract(self, text_processed_spacy):
        """
        Extracts all possible aspects - NER, NOUN and NOUN PHRASES,
        potentially other dictionary based aspects.

        Parameters
        ----------
        text_processed_spacy : dictionary
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
        tokens = text_processed_spacy['tokens']
        text = text_processed_spacy['raw_text']
        aspect_sequence = []
        aspect_sequence_main_encountered = False
        aspect_sequence_enabled = False
        concept_aspects = {}
        aspects = []

        # 1. look for NER examples
        if self.is_ner:
            aspects = text_processed_spacy['entities']

        # 2. NOUN and NOUN phrases
        for idx, token in enumerate(tokens):
            if self._is_interesting_main(token) and len(token) > 1:
                if not token['is_stop']:
                    aspect_sequence.append(token['lemma'])
                aspect_sequence_enabled = True
                aspect_sequence_main_encountered = True
            else:
                if aspect_sequence_enabled and aspect_sequence_main_encountered:
                    aspect = ' '.join(aspect_sequence)
                    if aspect not in aspects:
                        aspects.append(aspect)
                aspect_sequence_main_encountered = False
                aspect_sequence_enabled = False
                aspect_sequence = []

        if aspect_sequence_enabled and aspect_sequence_main_encountered:
            aspects.append(' '.join(aspect_sequence))

        # lower case every aspect and only longer than 1
        aspects = [x.strip().lower() for x in aspects if x not in self.aspects_to_skip and x != '']

        # 3. senticnet
        if settings.SENTIC_ASPECTS:
            sentic = Sentic()
            sentic_aspects = {}
            for aspect in aspects:
                aspect = aspect.replace(' ', '_')
                sentic_aspects[aspect] = sentic.get_semantic_concept_by_concept(
                    aspect, settings.SENTIC_EXACT_MATCH_CONCEPTS)
            concept_aspects['sentic'] = sentic_aspects

        # 4. ConceptNet.io
        if settings.CONCEPTNET_IO_ASPECTS:
            conceptnet_aspects = defaultdict(list)
            for aspect in aspects:
                if aspect not in self.conceptnet_io:
                    conceptnet_aspects[aspect] = self.conceptnet_io[aspects]
                else:
                    log.debug('We have already stored this concept: {}'.format(aspect))
                    conceptnet_aspects[aspect] = self.conceptnet_io[aspect]
            concept_aspects['conceptnet_io'] = conceptnet_aspects

        # 5. keyword extraction
        if text:
            keyword_aspects = {'rake': self.Rake.run(text)}
        else:
            keyword_aspects = {'rake': [(None, None)]}

        return aspects, concept_aspects, keyword_aspects
