import re
from datetime import datetime
from typing import Iterable

import jellyfish
from tqdm import tqdm

from aspects.utilities.common_nlp import load_spacy


class Preprocesser(object):
    def __init__(self):
        self.nlp = load_spacy()

    def preprocess(self, text):
        # preprocessing tekstu do analizy
        text = text.strip()
        # text = text.lower() # for NER it's better to have use upper/lower case
        text = text.decode('utf8')

        # FIXME: temporary cleaning of char that are specific to RQT trees
        # FIXME: this was the problem with word with q in the end!!
        text = re.sub('^U.', '', text)
        text = re.sub('q . <P>$', '', text)
        text = re.sub(' . <s>$', '', text)
        text = re.sub('\r\n$', '', text)

        # analiza spacy
        doc = self.nlp(text)

        # FIXME: move this to aspect extractor
        ner_types = [u'PERSON', u'GPE', u'ORG', u'PRODUCT', u'FAC', u'LOC']
        entities = [ent.lemma_ for ent in doc.ents if ent.label_ in ner_types]
        result = {'raw_text': doc.text, 'tokens': [], 'entities': entities}

        # TODO: it's better to use spacy objects than saving it to dictionary
        # zapisanie wyników dla każdego tokena tekstu
        for idx, token in enumerate(doc):
            # skip tokens with length lower than 2
            if len(token.string.strip()) > 1:
                token_info = {'text': token.orth_, 'pos': token.pos_,
                              'lemma': token.lemma_, 'is_stop': token.is_stop}
                result['tokens'].append(token_info)
                # else:
                #     print('Strange potential aspect: {}'.format(token))

        return result

    def lemmatize(self, aspect_text: str) -> str:
        return ' '.join([token.lemma_ for token in self.nlp(aspect_text)])

    def get_most_similar_text_pairs(self, texts: Iterable[str], threshold: int = 0.75):
        """ Similarity based on Jaro-Wrinkler distance """
        print(f'{datetime.now()} for #{len(texts)} of texts and #{len(texts) * len(texts)} of pairs')
        similar_pairs = {}
        for text_1 in tqdm(set(texts)):
            for text_2 in set(texts):
                jaro = jellyfish.jaro_winkler(text_1, text_2)

                # lemma with spacy
                text_1 = self.lemmatize(text_1)
                text_2 = self.lemmatize(text_2)

                texts_pair = f'{text_1} ## {text_2}'
                texts_pair_inverse = f'{text_2} ## {text_1}'
                if text_1 != text_2 and jellyfish.jaro_winkler(
                        text_1, text_2) > threshold and texts_pair_inverse not in similar_pairs:
                    similar_pairs[texts_pair] = jaro
        return similar_pairs
