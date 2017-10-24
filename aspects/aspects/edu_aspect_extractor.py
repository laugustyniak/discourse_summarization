# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from aspects.aspects.aspect_extractor import AspectExtractor


class EDUAspectExtractor(object):
    def __init__(self):
        self.extractor = AspectExtractor()

    def extract(self, edu, n_doc):
        """
        Extract aspects from edu document.

        Parameters
        ----------
        edu : dict
            Preprocessed string with tokens, ner and etc.

        n_doc : int
            Document of document processed.

        Returns
        -------
        aspects : dict

        concept_aspects : dict

        keyword_aspects : dict

        """
        return self.extractor.extract(edu, n_doc)
