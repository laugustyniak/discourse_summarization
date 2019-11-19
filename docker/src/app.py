import sys

from flask import Flask, request, jsonify
from flask_restful import Api

sys.path.append('/app/rhetorical_parser/src')

from parse import DiscourseParser

app = Flask(__name__)
api = Api(app)


@app.route('/api/rst/parse', methods=['POST'])
def extract_aspects():
    text = request.json['text']
    response = {
        'tree': parser.parse(text)
    }
    return jsonify(response)


if __name__ == '__main__':
    parser = DiscourseParser()
    app.run(debug=True, host='0.0.0.0')
