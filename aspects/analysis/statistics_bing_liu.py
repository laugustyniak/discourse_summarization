from collections import namedtuple
from glob import glob
from os.path import basename

import matplotlib.pyplot as plt
import pandas as pd

all_reviews_path = '../../aspects/data/aspects/Reviews-9-products/'
reviews_paths = glob(all_reviews_path + '*')

AspectSentiment = namedtuple('AspectSentiment', 'aspect, sentiment')
DatasetSize = namedtuple('DatasetSize', 'dataset, size')


def parse_reviews(reviews_path: str) -> str:
    with open(reviews_path, 'r') as review_file:
        for line in review_file:
            yield line


def load_reviews(reviews_path: str) -> pd.DataFrame:
    return pd.read_csv(reviews_path, sep='##', names=['aspects', 'text'])


def get_aspects(reviews_path: str) -> pd.DataFrame:
    all_aspects = []
    for aspect_str in list(load_reviews(reviews_path).dropna('index').aspects):
        aspects = aspect_str.split(',')
        for aspect in [a for a in aspects if a]:
            all_aspects.append(get_sentiment_from_aspect_sentiment_text(aspect))
    return pd.DataFrame(all_aspects)


def get_sentiment_from_aspect_sentiment_text(aspect_with_sentiment: str) -> AspectSentiment:
    aspect_with_sentiment = aspect_with_sentiment.strip()
    aspect_with_sentiment = aspect_with_sentiment.replace('[u]', '').replace('[s]', '').replace('[p]', '')
    aspect_with_sentiment = aspect_with_sentiment.replace('[cs]', '')
    aspect_with_sentiment = aspect_with_sentiment.replace('{', '[').replace('}', ']')

    aspect_splitted = aspect_with_sentiment.split('[')
    aspect = aspect_splitted[0]
    try:
        sentiment_str = aspect_splitted[1].replace(']', '')
    except IndexError:
        print(f'Error with: {aspect_with_sentiment}')
        raise Exception
    if '-' in sentiment_str:
        if len(sentiment_str) > 1:
            sentiment = int(sentiment_str)
        else:
            sentiment = -1
    elif '+' in sentiment_str:
        if len(sentiment_str) > 1:
            sentiment = int(sentiment_str)
        else:
            sentiment = 1
    else:
        sentiment = int(sentiment_str)
    return AspectSentiment(aspect=aspect.lower(), sentiment=sentiment)


def draw_aspect_distribution(reviews_path: str):
    df = get_aspects(reviews_path)
    plt.figure(figsize=(20, 8))
    plt.title(f'Aspect distribution for {basename(reviews_path)}')
    counts = df.aspect.value_counts()
    # take only aspects that appeared at least once 
    counts[counts > 1].plot(kind='bar')


def get_number_of_reviews(reviews_path: str) -> int:
    return load_reviews(reviews_path).shape[0]


def get_number_of_reviews_with_aspects(reviews_path: str) -> int:
    return load_reviews(reviews_path).dropna('index').shape[0]


def get_datasets_sizes(reviews_paths: str) -> pd.DataFrame:
    return pd.DataFrame([
        DatasetSize(dataset=basename(reviews_path), size=get_number_of_reviews(reviews_path))
        for reviews_path in reviews_paths
    ])


for reviews_path in reviews_paths:
    get_aspects(reviews_path)
    # get_datasets_sizes(reviews_paths).plot(kind='bar', x='dataset')
