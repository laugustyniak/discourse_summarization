import requests

from aspects.utilities.settings import SENTIMENT_DOCKER_URL


class SentimentClient:

    def __init__(self, url=None, json_request_key=None, json_response_key=None):
        self.url = url or {SENTIMENT_DOCKER_URL}
        self.json_request_key = json_request_key or 'text'

    def parse(self, text):
        return requests.post(self.url, json={self.json_request_key: text}).json()
