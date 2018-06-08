import logging
from collections import defaultdict

import RAKE
from gensim.summarization import keywords

from aspects.enrichments.conceptnets import load_sentic, load_conceptnet_io, get_semantic_concept_by_concept
from aspects.utilities import common_nlp
from aspects.utilities import settings

log = logging.getLogger(__name__)

nlp = common_nlp.load_spacy()


class AspectExtractor(object):
    """ Extract aspects from EDU. """

    def __init__(
            self, ner_types={u'PERSON', u'GPE', u'ORG', u'PRODUCT', u'FAC', u'LOC'}, aspects_to_skip=None, is_ner=True):
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
            self.aspects_to_skip = common_nlp.ASPECTS_TO_SKIP

        if ner_types is None:
            self.ner_types = settings.NER_TYPES
        else:
            self.ner_types = ner_types

        self.Rake = RAKE.Rake(RAKE.SmartStopList())

        if settings.CONCEPTNET_IO_ASPECTS:
            self.conceptnet_io = load_conceptnet_io()

    def _is_interesting_main(self, token):
        return token.pos_ == 'NOUN'

    def _is_interesting_addition(self, token):
        return token.pos_ == 'ADV' or token.pos_ == 'NUM' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ'

    def extract(self, text):
        """
        Extracts all possible aspects - NER, NOUN and NOUN PHRASES,
        potentially other dictionary based aspects.

        Parameters
        ----------
        text : dictionary
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
        concept_aspects = {}
        aspects = self.extract_noun_and_noun_phrases(text)

        if self.is_ner:
            aspects += [ent.lemma_ for ent in nlp(text).ents if ent.label_ in self.ner_types]

        # lower case every aspect and only longer than 1
        aspects = [x.strip().lower() for x in aspects if x not in self.aspects_to_skip and x != '']

        # 3. senticnet
        if settings.SENTIC_ASPECTS:
            concept_aspects['sentic'] = self.extract_concepts_from_sentic(aspects)

        # 4. ConceptNet.io
        if settings.CONCEPTNET_IO_ASPECTS:
            concept_aspects['conceptnet_io'] = self.extract_concept_from_conceptnet_io(aspects)

        return aspects, concept_aspects, self.extract_keywords(text)

    def extract_noun_and_noun_phrases(self, text):
        aspects = []
        aspect_sequence = []
        aspect_sequence_main_encountered = False
        aspect_sequence_enabled = False
        for token in nlp(text):
            if self._is_interesting_main(token):
                if not token.is_stop and len(aspect_sequence) < 4:
                    aspect_sequence.append(token.lemma_)
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
        return aspects

    def extract_concept_from_conceptnet_io(self, aspects):
        conceptnet_aspects = defaultdict(list)
        for aspect in aspects:
            if aspect not in self.conceptnet_io:
                conceptnet_aspects[aspect] += self.conceptnet_io[aspect]
        return conceptnet_aspects

    def extract_concepts_from_sentic(self, aspects):
        sentic_df = load_sentic()
        sentic_aspects = {}
        for aspect in aspects:
            aspect = aspect.replace(' ', '_')
            sentic_aspects[aspect] = get_semantic_concept_by_concept(
                sentic_df, aspect, settings.SENTIC_EXACT_MATCH_CONCEPTS)
        return sentic_aspects

    def extract_keywords(self, text):
        if text:
            return {
                'rake': self.Rake.run(text),
                'text_rank': keywords(text)
            }
        else:
            return {
                'rake': [(None, None)],
                'text_rank': []
            }
