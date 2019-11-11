import uuid
from shutil import rmtree

from flask import Flask, request, jsonify
from flask_restful import Api

from parse import DiscourseParser

app = Flask(__name__)
api = Api(app)

TREE_TEMP_DIR = '/tmp/'


@app.route('/api/rst/parse', methods=['POST'])
def extract_aspects():
    tree_temp_name = str(uuid.uuid4())
    text = request.json['text']
    response = {
        'tree': parser.parse(tree_temp_name, parse_text=text)
    }
    rmtree(TREE_TEMP_DIR + tree_temp_name)
    return jsonify(response)


if __name__ == '__main__':
    parser = DiscourseParser(output_dir=TREE_TEMP_DIR)
    app.run(debug=True, host='0.0.0.0')
