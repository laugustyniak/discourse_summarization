from typing import List

import requests

from aspects.utilities.settings import SENTIMENT_DOCKER_URL


class BiLSTMModel:
    def __init__(self, url=None, json_request_key=None, json_response_key=None):
        self.url = url or SENTIMENT_DOCKER_URL
        self.json_request_key = json_request_key or "text"
        self.json_response_key = json_response_key or "sentiment"

    def analyse(self, text):
        return requests.post(self.url, json={self.json_request_key: text}).json()

    def get_sentiments(self, texts: List[str]) -> List[float]:
        return [self.analyse(text)[self.json_response_key] for text in texts]
