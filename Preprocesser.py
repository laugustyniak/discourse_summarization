# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import spacy
import re
from textlytics.sentiment import DocumentPreprocessor


class Preprocesser:
    def __init__(self):
        print 'Preprocesser: initializing'
        # load spacy with parsers and entities, it will be useful in next steps of analysis
        self.nlp = spacy.load('en')
        self.dp = DocumentPreprocessor()
        print 'Preprocesser: initialized'

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

        # analiza spacy
        doc = self.nlp(text)

        # FIXME: move this to aspect extractor
        ner_types = [u'PERSON', u'GPE', u'ORG', u'PRODUCT', u'FAC', u'LOC']
        entities = [ent.lemma_ for ent in doc.ents if ent.label_ in ner_types]
        result = {'raw_text': doc.text, 'tokens': [], 'entities': entities}

        # TODO: it's better to use spacy objects than saving it to dictionary
        # zapisanie wyników dla każdego tokena tekstu
        for idx, token in enumerate(doc):
            token_info = {'text': token.orth_, 'pos': token.pos_, 'lemma': token.lemma_,
                          'is_stop': token.is_stop}

            result['tokens'].append(token_info)

        return result
