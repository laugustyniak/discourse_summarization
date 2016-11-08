# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda
import re
import spacy
import morfeusz2


class Preprocesser:
    def __init__(self, lang, punctuation='\'"#&$%`~', words_to_skip=['ok']):
        print 'Preprocesser: initializing'
        self.nlp = spacy.en.English(parser=False, entity=False)
        print 'Preprocesser: initialized'
        self.punctuation = punctuation
        self.words_to_skip = words_to_skip
        self.lang = lang

    def preprocess(self, text):

        result = {'raw_text': text, 'tokens': []}

        # preprocessing tekstu do analizy
        text = text.strip()
        text = text.lower()
        text = text.decode('utf8')

        if self.lang in ['en']:
            doc = self.nlp(text, entity=False, parse=False)
            for token in doc:
                token_info = {'text': token.orth_,
                              'pos': token.pos_,
                              'lemma': token.lemma_,
                              'is_stop': token.is_stop}
                result['tokens'].append(token_info)
        elif self.lang in ['pl']:
            m = morfeusz2.Morfeusz()
            for token in text.split():
                token_info = {'text': self.get_morfeusz_lemma(m, token),
                              'pos': self.get_morfeusz_nouns(m, token),
                              'lemma': self.get_morfeusz_lemma(m, token),
                              'is_stop': False}
                result['tokens'].append(token_info)
            print result
        else:
            raise Exception("Wrong language in preprocessing step")
        return result

    def get_morfeusz_nouns(self, m, text):
        return 'NOUN' if m.analyse(text)[0][2][2].split(':')[0] == 'subst' else ''

    def get_morfeusz_lemma(self, m, text):
        lemma = m.analyse(text)[0][2][1].encode('utf-8').split(':')[0]
        # useful because of 'ok' would be lematize to 'oko;, and others
        if lemma in self.words_to_skip:
            lemma = text
        return lemma

    def remove_punctuation(self, text):
        regex = re.compile('[%s]' % re.escape(self.punctuation))
        text = regex.sub(' ', text)
        return ' '.join(text.split())
