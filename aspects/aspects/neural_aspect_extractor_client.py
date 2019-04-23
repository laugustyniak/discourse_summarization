import requests
from flask.json import jsonify


class NeuralAspectExtractorClient:

    def extract(self, text):
        return jsonify(requests.post(
            "http://localhost:5000/api/aspects/extract",
            data={'text': text}
        ))['aspects']
