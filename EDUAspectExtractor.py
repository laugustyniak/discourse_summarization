# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from AspectExtractor import AspectExtractor


class EDUAspectExtractor(object):
    def __init__(self):
        self.extractor = AspectExtractor()

    def extract(self, edu):
        """
        Extract aspects and apsect_concepts
        :param edu:
        :return:
        """
        return self.extractor.extract(edu)

    def get_aspects_in_document(self, documents_aspects, aspects):

        if documents_aspects is None:
            return aspects
        else:
            doc_asp = documents_aspects + aspects

            # wymuszamy unikalnosc
            doc_asp = list(set(doc_asp))

            return doc_asp
