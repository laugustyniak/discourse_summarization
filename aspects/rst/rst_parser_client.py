import requests


class RSTParserClient:

    def __init__(self, url=None, json_request_key=None, json_response_key=None):
        # self.url = url or "http://localhost:5000/api/rst/parse"
        self.url = url or "http://oxygen.engine.kdm.wcss.pl:5000/api/rst/parse"
        self.json_request_key = json_request_key or 'text'
        self.json_response_key = json_response_key or 'tree'

    def parse(self, text):
        return requests.post(
            self.url,
            json={
                self.json_request_key: text
            }
        ).json()[self.json_response_key]
