from fastapi import FastAPI
from flair.data import Sentence
from flair.models import TextClassifier
from pydantic import BaseModel

app = FastAPI()
classifier = TextClassifier.load('en-sentiment')


class Request(BaseModel):
    text: str


@app.post('/api/sentiment/')
async def sentiment(request: Request):
    sentence = Sentence(request.text)
    classifier.predict(sentence)
    label = sentence.labels[0]
    return {
        'value': label.value,
        'sentiment': label.score
    }
