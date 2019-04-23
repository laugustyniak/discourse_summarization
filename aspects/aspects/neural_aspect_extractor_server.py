import tensorflow as tf
from flask import Flask, request, jsonify

from aspects.aspects.neural_aspect_extractor import NeuralAspectExtractor

GRAPH = tf.get_default_graph()
app = Flask(__name__)


@app.route('/api/aspects/extract', methods=['POST'])
def extract_aspects():
    message = request.json['message']
    print(type(message))
    print(message)
    # get graph to load tf session properly
    with GRAPH.as_default():
        response = {
            'aspects': model.extract(message)
        }
    return jsonify(response)


if __name__ == '__main__':
    # Model is loaded when the API is launched
    model = NeuralAspectExtractor()
    app.run(debug=True)