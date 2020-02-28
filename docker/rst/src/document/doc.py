from base_representation import BaseRepresentation


class Document(BaseRepresentation):
    def __init__(self):
        BaseRepresentation.__init__(self)

        self.preprocessed = False
        self.segmented = False
        self.parsed = False

        self.sentences = []
        self.edus = None
        self.cuts = None
        self.edu_word_segmentation = None
        self.start_edu = None
        self.end_edu = None
        self.discourse_tree = None

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def preprocess(self, text, preprocesser):
        preprocesser.preprocess(text, self)
        self.preprocessed = True

    def get_bottom_level_constituents(self):
        constituents = []
        for sentence in self.sentences:
            assert sentence.discourse_tree and len(sentence.constituents) == 1
            constituents.append(sentence.constituents[0])

        return constituents
