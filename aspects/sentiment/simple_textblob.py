from typing import List

from textblob import TextBlob


def analyze(texts: List[str]) -> List[float]:
    return [TextBlob(text).sentiment.polarity for text in texts]
