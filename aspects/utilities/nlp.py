from repoze.lru import lru_cache

import spacy


@lru_cache(None)
def load_spacy(model='en'):
    return spacy.load(model)
