import logging
import re
from datetime import datetime

import jellyfish
from tqdm import tqdm

from aspects.utilities.common_nlp import load_spacy

log = logging.getLogger(__name__)

try:
    from wordfreq import zipf_frequency
except:
    log.error('No wordfreq lib installed')


class Preprocesser(object):
    def __init__(self):
        self.nlp = load_spacy()

    def preprocess(self, text):
        text = text.strip()

        # FIXME: temporary cleaning of char that are specific to RQT trees
        # FIXME: this was the problem with word with q in the end!!
        text = re.sub('^U.', '', text)
        text = re.sub('q . <P>$', '', text)
        text = re.sub(' . <s>$', '', text)
        text = re.sub('\r\n$', '', text)
        return text

    def lemmatize(self, aspect_text):
        return ' '.join([token.lemma_ for token in self.nlp(aspect_text)])

    def get_most_similar_text_pairs(self, texts, threshold=0.75):
        """ Similarity based on Jaro-Wrinkler distance """
        print('{} for #{} of texts and #{} of pairs'.format(datetime.now(), len(texts), len(texts) * len(texts)))
        similar_pairs = {}
        for text_1 in tqdm(set(texts)):
            for text_2 in set(texts):
                jaro = jellyfish.jaro_winkler(text_1, text_2)

                # lemma with spacy
                text_1 = self.lemmatize(text_1)
                text_2 = self.lemmatize(text_2)

                texts_pair = '{} ## {}'.format(text_1, text_2)
                texts_pair_inverse = '{} ## {}'.format(text_2, text_1)
                if text_1 != text_2 and jellyfish.jaro_winkler(
                        text_1, text_2) > threshold and texts_pair_inverse not in similar_pairs:
                    similar_pairs[texts_pair] = jaro
        return similar_pairs

    def filter_words_by_zipf(self, words, max_zipf_freq=4.0):
        """ Zipf equal to 4 is boarder between common [zipf higher than 4] and uncommon words [otherwise]. """
        return [
            word
            for word
            in words
            if zipf_frequency(word, 'en') < max_zipf_freq
        ]
