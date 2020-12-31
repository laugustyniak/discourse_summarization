import requests
import uuid
from aspects.utilities.settings import RST_PARSER_DOCKER_URL


class RSTParserClient:
    def __init__(self, url=None):
        self.url = url or RST_PARSER_DOCKER_URL

    def parse(self, text: str) -> str:
        files = {"input": (f"{str(uuid.uuid4())}.txt", text)}
        response = requests.post(self.url, files=files)
        return response.text
