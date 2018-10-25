import logging
from collections import defaultdict

from aspects.enrichments.conceptnets import load_sentic, load_conceptnet_io, get_semantic_concept_by_concept
from aspects.utilities import common_nlp
from aspects.utilities import settings

log = logging.getLogger(__name__)

nlp = common_nlp.load_spacy()


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
        if ner_types is None:
            ner_types = {u'PERSON', u'GPE', u'ORG', u'PRODUCT', u'FAC', u'LOC'}
        self.is_ner = is_ner
        if aspects_to_skip is not None:
            self.aspects_to_skip = aspects_to_skip
        else:
            self.aspects_to_skip = common_nlp.ASPECTS_TO_SKIP

        if ner_types is None:
            self.ner_types = settings.NER_TYPES
        else:
            self.ner_types = ner_types

        self.aspects_word_ids = []

    def _is_interesting_addition(self, token):
        return token.pos_ == 'ADV' or token.pos_ == 'NUM' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ'

    def extract(self, text):
        """
        Extracts all possible aspects - NER, NOUN and NOUN PHRASES,
        potentially other dictionary based aspects.

        Parameters
        ----------
        text : str
            unicode string

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
        text = unicode(text)

        aspects, self.aspects_word_ids = self.extract_noun_and_noun_phrases(text)

        if self.is_ner:
            aspects += [ent for ent in nlp(text).ents if ent.label_ in self.ner_types]

        # lower case every aspect and only longer than 1
        aspects = [x.strip().lower() for x in aspects if x not in self.aspects_to_skip and x != '']

        if settings.SENTIC_ASPECTS:
            concept_aspects['sentic'] = self.extract_concepts_from_sentic(aspects)

        if settings.CONCEPTNET_IO_ASPECTS:
            concept_aspects['conceptnet_io'] = self.extract_concept_from_conceptnet_io(aspects)

        return aspects, concept_aspects, self.extract_keywords(text)

    def extract_noun_and_noun_phrases(self, text):
        aspects = []
        aspects_word_ids = []
        aspect_sequence = []
        aspect_sequence_main_encountered = False
        aspect_sequence_enabled = False
        for token_id, token in enumerate(nlp(text)):
            if token.pos_ == 'NOUN':
                if len(aspect_sequence) < 3:
                    aspect_sequence.append(token.lemma_)
                    aspects_word_ids.append(token_id)
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
        aspects = [aspect for aspect in aspects if aspect]
        return aspects, aspects_word_ids

    def _aspects_to_conll_format(self, text, aspects):
        # TODO: i ended here
        pass

    def extract_concept_from_conceptnet_io(self, aspects):
        conceptnet_io = load_conceptnet_io()
        conceptnet_aspects = defaultdict(list)
        for aspect in aspects:
            if aspect in conceptnet_io:
                conceptnet_aspects[aspect] += conceptnet_io[aspect]
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
        rake = common_nlp.load_rake()
        if text:
            try:
                # text_rank_keywords = keywords(text)
                text_rank_keywords = []
            except:
                log.error('Cant get keywords from text: {}'.format(text))
                text_rank_keywords = []

            return {
                'rake': rake.run(text),
                'text_rank': text_rank_keywords
            }
        else:
            return {
                'rake': [(None, None)],
                'text_rank': []
            }

    def extract_aspects_bilstm_crf_model(self):
        # TODO: implement
        pass
