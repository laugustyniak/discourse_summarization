import logging
import re
from datetime import datetime

import jellyfish
from tqdm import tqdm

from aspects.utilities.common_nlp import load_spacy

log = logging.getLogger(__name__)

nlp = load_spacy()


def preprocess(text):
    text = text.strip()
    return {'text': text}


def lemmatize(aspect_text):
    return ' '.join([token.lemma_ for token in nlp(aspect_text)])


def get_most_similar_text_pairs(texts, threshold=0.75):
    """ Similarity based on Jaro-Wrinkler distance """
    print('{} for #{} of texts and #{} of pairs'.format(datetime.now(), len(texts), len(texts) * len(texts)))
    similar_pairs = {}
    for text_1 in tqdm(set(texts)):
        for text_2 in set(texts):
            jaro = jellyfish.jaro_winkler(text_1, text_2)

            # lemma with spacy
            text_1 = lemmatize(text_1)
            text_2 = lemmatize(text_2)

            texts_pair = '{} ## {}'.format(text_1, text_2)
            texts_pair_inverse = '{} ## {}'.format(text_2, text_1)
            if text_1 != text_2 and jellyfish.jaro_winkler(
                    text_1, text_2) > threshold and texts_pair_inverse not in similar_pairs:
                similar_pairs[texts_pair] = jaro
    return similar_pairs
