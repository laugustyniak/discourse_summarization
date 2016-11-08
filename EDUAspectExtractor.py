# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from POSAspectExtractor import POSAspectExtractor


class EDUAspectExtractor():
    def __init__(self):
        self.extractor = POSAspectExtractor()

    def extract(self, EDU):

        aspects = self.extractor.extract(EDU)

        return aspects

    def getAspectsInDocument(self, documentsAspects, aspects):

        if documentsAspects is None:
            return aspects
        else:
            docAsp = documentsAspects + aspects

            # wymuszamy unikalnosc
            docAsp = list(set(docAsp))

            return docAsp
