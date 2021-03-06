import logging
from collections import defaultdict
from typing import List, Sequence, Dict

from aspects.aspects.neural_aspect_extractor_client import NeuralAspectExtractorClient
from aspects.enrichments.conceptnets import (
    load_sentic,
    load_conceptnet_io,
    get_semantic_concept_by_concept,
)
from aspects.utilities import common_nlp
from aspects.utilities import settings

log = logging.getLogger(__name__)

nlp = common_nlp.load_spacy()


class AspectExtractor:
    def __init__(
        self,
        ner_types=None,
        aspects_to_skip=None,
        is_ner=True,
        sentic=None,
        conceptnet=None,
    ):
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
            ner_types = {u"PERSON", u"GPE", u"ORG", u"PRODUCT", u"FAC", u"LOC"}
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

        if sentic is None:
            self.sentic = settings.SENTIC_ASPECTS
        else:
            self.sentic = sentic

        if conceptnet is None:
            self.conceptnet = settings.CONCEPTNET_IO_ASPECTS
        else:
            self.conceptnet = conceptnet

        self.neural_aspect_extractor_client = NeuralAspectExtractorClient()

    def extract_batch(self, texts: Sequence[str]) -> Sequence[List[str]]:
        return [self.extract(text) for text in texts]

    def extract(self, text: str) -> List[str]:
        aspects = self.neural_aspect_extractor_client.extract(text)

        if self.is_ner:
            aspects += [
                ent.text for ent in nlp(text).ents if ent.label_ in self.ner_types
            ]

        # lower case every aspect and only longer than 1
        return [
            x.strip().lower()
            for x in aspects
            if x not in self.aspects_to_skip and len(x) > 1
        ]

    def extract_concept_from_conceptnet_io(self, aspects: List) -> Dict:
        conceptnet_io = load_conceptnet_io()
        conceptnet_aspects = defaultdict(list)
        for aspect in aspects:
            if aspect in conceptnet_io:
                conceptnet_aspects[aspect] += conceptnet_io[aspect]
        return conceptnet_aspects

    def extract_concepts_from_sentic(self, aspects: List):
        sentic_df = load_sentic()
        sentic_aspects = {}
        for aspect in aspects:
            aspect = aspect.replace(" ", "_")
            sentic_aspects[aspect] = get_semantic_concept_by_concept(
                sentic_df, aspect, settings.SENTIC_EXACT_MATCH_CONCEPTS
            )
        return sentic_aspects

    def extract_concepts_batch(self, aspects: Sequence[List[str]]) -> List[Dict]:
        concepts = []
        for aspects_internal in aspects:
            concept_aspects = {}
            if self.sentic:
                concept_aspects["sentic"] = self.extract_concepts_from_sentic(
                    aspects_internal
                )
            if self.conceptnet:
                concept_aspects[
                    "conceptnet_io"
                ] = self.extract_concept_from_conceptnet_io(aspects_internal)
            concepts.append(concept_aspects)
        return concepts
