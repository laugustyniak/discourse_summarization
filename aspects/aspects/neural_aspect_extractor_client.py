import requests


class NeuralAspectExtractorClient:

    def __init__(self, url=None, json_request_key=None, json_response_key=None):
        self.url = url or "http://localhost:5000/api/aspects/extract"
        self.json_request_key = json_request_key or 'text'
        self.json_response_key = json_response_key or 'aspects'

    def extract(self, text):
        return requests.post(
            self.url,
            json={
                self.json_request_key: text
            }
        ).json()[self.json_response_key]
