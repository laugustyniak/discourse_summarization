from aspects.aspects.aspect_extractor import AspectExtractor


class EDUAspectExtractor(object):
    def __init__(self):
        self.extractor = AspectExtractor()

    def extract(self, edu):
        """
        Extract aspects from edu document.

        Parameters
        ----------
        edu : dict
            Preprocessed string with tokens, ner and etc.


        Returns
        -------
        aspects : dict

        concept_aspects : dict

        keyword_aspects : dict

        """
        return self.extractor.extract(edu['text'])
