class BaseRepresentation:
    def __init__(self):
        self.constituents = []
        self.constituent_scores = []

    def prepare_parsing(self):
        self.constituents = self.get_bottom_level_constituents()
