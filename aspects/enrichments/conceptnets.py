import logging
import pickle
import pandas as pd
from os.path import join

from data.sentic.senticnet4.senticnet4 import senticnet
from aspects.project_path import CNIO_PATH

log = logging.getLogger(__name__)


class Sentic(object):
    def __init__(self):
        self.sentic_df = pd.DataFrame.from_dict(senticnet, orient='index')
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


class ConceptNetIO(object):
    def __init__(self, f_name_concepts_io='conceptnet_io.pkl'):
        self.concepts_io = {}
        self.concepts_io_path = join(CNIO_PATH, f_name_concepts_io)

    def save_cnio(self):
        log.info('ConceptNet.io temp files will be stored in: {}'.format(
            self.concepts_io_path))
        with open(self.concepts_io_path, 'wb') as f:
            pickle.dump(self.concepts_io, f)
            log.info('ConceptNet.io temp file loaded correctly')

    def load_cnio(self):
        try:
            log.info('ConceptNet.io temp files will be load from: {}'.format(
                self.concepts_io_path))
            with open(self.concepts_io_path, 'rb') as f:
                self.concepts_io = pickle.load(f)
        except (IOError, EOFError) as err:
            log.error(str(err))
            raise Exception()
