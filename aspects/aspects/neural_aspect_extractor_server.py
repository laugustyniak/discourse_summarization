from flask import Flask, request, jsonify, abort

from aspects.aspects.neural_aspect_extractor import NeuralAspectExtractor

app = Flask(__name__)


@app.route('/api/aspects/extract', methods=['POST'])
def extract_aspects():
    if not request.json or 'message' not in request.json:
        abort(400)
    message = request.json['message']
    response = {
        'aspects': model.extract(message)
    }
    return jsonify(response)


if __name__ == '__main__':
    # Model is loaded when the API is launched
    model = NeuralAspectExtractor()
    app.run(debug=True)
