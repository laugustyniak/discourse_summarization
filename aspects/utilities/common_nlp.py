from functools import lru_cache

import spacy


@lru_cache(None)
def load_spacy(model: str = 'en_core_web_sm'):
    return spacy.load(model)
