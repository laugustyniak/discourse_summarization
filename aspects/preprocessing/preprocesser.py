# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import spacy
import re


class Preprocesser(object):
    def __init__(self):
        # load spacy with parsers and entities,
        # it will be useful in next steps of analysis
        self.nlp = spacy.load('en')

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
