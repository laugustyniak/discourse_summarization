import logging

from sklearn.externals import joblib

log = logging.getLogger(__name__)


class LogisticRegressionSentimentAnalyzer(object):

    def __init__(self, model_path=None):
        log.info('Sentiment Analyzer - status: initializing')

        if model_path is None:
            # model_path = 'data/models/' \
            #              'Pipeline-LogisticRegression-' \
            #              'CountVectorizer-n_grams_1_2-' \
            #              'stars-1-3-5-' \
            #              '10-domains.pkl'
            # smaller model as default - useful for testing
            model_path = 'data/models/' \
                         'Pipeline-LogisticRegression-' \
                         'CountVectorizer-n_grams_1_2-' \
                         'stars-1-3-5-' \
                         'reviews_Apps_for_Android-500000-balanced.pkl'
            log.info('Default model: {} will loaded.'.format(model_path))
        self.model = joblib.load(model_path)

        log.info('Sentiment Analyzer - status: initialized')

    def analyze(self, text):
        return self.model.predict([text])
