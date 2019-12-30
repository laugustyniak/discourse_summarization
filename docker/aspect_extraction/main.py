import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

from neural_aspect_extractor import NeuralAspectExtractor

app = FastAPI()

GRAPH = tf.get_default_graph()
model = NeuralAspectExtractor()


class Request(BaseModel):
    text: str


@app.post('/api/aspects/')
async def sentiment(request: Request):
    # get graph to load tf session properly
    with GRAPH.as_default():
        return {
            'aspects': model.extract(request.text)
        }
