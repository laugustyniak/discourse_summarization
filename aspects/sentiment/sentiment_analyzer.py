import logging

from sklearn.externals import joblib

log = logging.getLogger(__name__)
from aspects.utilities import settings


class LogisticRegressionSentimentAnalyzer(object):

    def __init__(self, model_path=None):
        log.info('Sentiment Analyzer - status: initializing')

        if model_path is None:
            log.info('Defaul sentiment model will be loaded: {}'.format(settings.SENTIMENT_MODEL_PROD.as_posix()))
            self.model = joblib.load(settings.SENTIMENT_MODEL_PROD.as_posix())
        else:
            self.model = joblib.load(model_path)

        log.info('Sentiment Analyzer - status: initialized')

    def analyze(self, text):
        return self.model.predict([text])
