import sys

from flask import Flask, request, jsonify, abort
from flask_restful import Api

sys.path.append('/app/rhetorical_parser/src')

from parse import DiscourseParser

app = Flask(__name__)
api = Api(app)

parser = DiscourseParser()


@app.route('/api/rst/parse', methods=['POST'])
def extract_aspects():
    if not request.json or 'text' not in request.json:
        abort(400)
    text = request.json['text']
    response = {
        'tree': parser.parse(text)
    }
    return jsonify(response)
