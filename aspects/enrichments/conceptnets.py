import logging
import pickle

import pandas as pd
from repoze.lru import lru_cache

from aspects.data.sentic.senticnet4 import senticnet4
from aspects.utilities import settings

log = logging.getLogger(__name__)


class Sentic(object):
    def __init__(self):
        self.sentic_df = pd.DataFrame.from_dict(senticnet4, orient='index')
        self.sentic_df.columns = ['pleasantness', 'attention', 'sensivity',
                                  'aptitude', 'modtag1', 'modtag2',
                                  'polarity_value', 'polarity_intensity',
                                  'semantics1', 'semantics2', 'semantics3',
                                  'semantics4', 'semantics5']

    def get_concept_from_senticnet_by_partname(self, partname):
        """
        Get part of data frame with all concept's data related to partname
        concept partname could be regex
        :param partname: concept that will be filtered
        :return: Data Frame with concepts
        """
        return self.sentic_df[
            self.sentic_df.index.str.contains(partname)].sort_index()

    def get_semantic_concept_by_concept(self, partname, exact_match=False):
        """
        Get concept and list of related concepts related to partname concept,
        partname could be regex

        :param exact_match: bool
            Do we want to find exactly same concepts? Otherwise we will get
            all concepts with even substring of partname concept.
        :param partname: str
            Concept that will be filtered.

        :return: dict
            Concepts dictionary key: concept name,
            values: list of related concepts
        """
        concepts = {}
        semantic_col_name = ['semantics1', 'semantics2', 'semantics3',
                             'semantics4', 'semantics5']
        if exact_match:
            df = self.sentic_df[
                self.sentic_df.index == partname].sort_index()
        else:
            df = self.sentic_df[
                self.sentic_df.index.str.contains(partname)].sort_index()
        for row in df.iterrows():
            related_concepts = []
            for col in semantic_col_name:
                related_concepts.append(row[1][col])
            concepts[row[0]] = related_concepts
        return concepts


@lru_cache(maxsize=None)
def load_conceptnet_io():
    log.info('ConceptNet.io temp files will be load from: {}'.format(settings.CONCEPTNET_IO_PKL))
    with open(settings.CONCEPTNET_IO_PKL.as_posix(), 'rb') as f:
        return pickle.load(f)
