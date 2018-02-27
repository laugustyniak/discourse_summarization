import logging

from repoze.lru import lru_cache
from sklearn.externals import joblib

from aspects.configs.sentiment_config import SENTIMENT_MODEL_PATH

log = logging.getLogger(__name__)


@lru_cache(None)
def load_sentiment_analyzer(model_path=None):
    log.info('Sentiment Analyzer - status: initializing')
    if model_path is None:
        log.info('Defaul sentiment model will be loaded: {}'.format(SENTIMENT_MODEL_PATH))
        model = joblib.load(SENTIMENT_MODEL_PATH)
    else:
        model = joblib.load(model_path)
    log.info('Sentiment Analyzer - status: initialized')
    return model
