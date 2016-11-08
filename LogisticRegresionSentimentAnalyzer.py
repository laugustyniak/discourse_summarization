# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from sklearn.externals import joblib
import os


class LogisticRegresionSentimentAnalyzer:
    def __init__(self, model_path=None):
        print 'SentimentAnalyzer: initializing'

        if model_path is None:
            self.model = joblib.load(os.path.dirname(os.path.realpath(
                __file__)) + '/Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5-Electronics.pkl')
        else:
            self.model = joblib.load(model_path)

        print 'SentimentAnalyzer: initialized'

    def analyze(self, text):
        return self.model.predict([text])
