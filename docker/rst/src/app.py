import sys

from flask import Flask, request, jsonify, abort
from flask_restful import Api

sys.path.append('/app/rhetorical_parser/src')

from parse import DiscourseParser

app = Flask(__name__)
api = Api(app)


@app.route('/api/rst/parse', methods=['POST'])
def extract_aspects():
    if not request.json or not 'text' in request.json:
        abort(400)
    text = request.json['text']
    response = {
        'tree': parser.parse(text)
    }
    return jsonify(response)


if __name__ == '__main__':
    parser = DiscourseParser()
    app.run(host='0.0.0.0')
