# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from aspects.aspects.aspect_extractor import AspectExtractor


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
