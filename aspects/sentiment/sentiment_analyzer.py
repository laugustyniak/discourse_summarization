import logging

from sklearn.externals import joblib

log = logging.getLogger(__name__)
from aspects.configs.sentiment_config import SENTIMENT_MODEL_PATH


class LogisticRegressionSentimentAnalyzer(object):

    def __init__(self, model_path=None):
        log.info('Sentiment Analyzer - status: initializing')

        if model_path is None:
            log.info('Defaul sentiment model will be loaded: {}'.format(
                SENTIMENT_MODEL_PATH))
            self.model = joblib.load(SENTIMENT_MODEL_PATH)
        else:
            self.model = joblib.load(model_path)

        log.info('Sentiment Analyzer - status: initialized')

    def analyze(self, text):
        return self.model.predict([text])
