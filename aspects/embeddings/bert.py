import numpy as np
import spacy
from bert_embedding import BertEmbedding

nlp = spacy.load('en', disable=['parser', 'tagger'])


class BertWrapper:

    def __init__(self, bert_embedding: BertEmbedding = None):
        if bert_embedding is None:
            self.bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')

    def get_averaged_vector(self, text: str):
        return np.mean([
            self.bert_embedding([text])[0][1][0]
            for token
            in nlp(text)
        ], axis=0)
