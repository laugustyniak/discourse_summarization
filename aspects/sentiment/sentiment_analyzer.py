# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from sklearn.externals import joblib


class LogisticRegressionSentimentAnalyzer:
    def __init__(self, model_path=None):
        print 'SentimentAnalyzer: initializing'

        if model_path is None:
            # self.model = joblib.load('data/models/Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5-10-domains.pkl')
            # smaller model as default - useful for testing
            self.model = joblib.load('data/models/Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5-reviews_Apps_for_Android-500000-balanced.pkl')
        else:
            self.model = joblib.load(model_path)

        print 'SentimentAnalyzer: initialized'

    def analyze(self, text):
        return self.model.predict([text])
