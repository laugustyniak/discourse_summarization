from uuid import uuid4

import requests
from requests import ReadTimeout

from aspects.utilities.settings import RST_PARSER_DOCKER_URL


class RSTParserClient:
    def __init__(self, url=None):
        self.url = url or RST_PARSER_DOCKER_URL

    def parse(self, text: str) -> str:
        files = {"input": (f"{str(uuid4())}.txt", text)}
        try:
            response = requests.post(self.url, files=files, timeout=30)
            return (
                response.content.decode('utf-8').replace('\\n', '\n')
            )
        except ReadTimeout:
            return ""
