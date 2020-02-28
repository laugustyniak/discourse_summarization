from flask import Flask, request, jsonify, abort

from parse import DiscourseParser

app = Flask(__name__)
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
