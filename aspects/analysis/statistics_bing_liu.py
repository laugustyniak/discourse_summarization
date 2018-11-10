from collections import namedtuple
from os.path import basename

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List

from aspects.utilities import settings

AspectSentiment = namedtuple('AspectSentiment', 'aspect, sentiment')
DatasetSize = namedtuple('DatasetSize', 'dataset, size, reviews_word_average, aspect_coverage')


def load_reviews(reviews_path: str) -> pd.DataFrame:
    df = pd.read_csv(reviews_path, sep='##', names=['aspects', 'text'], engine='python')
    df = df[df.text.apply(lambda text: bing_liu_line_filter(text))]
    return df


def bing_liu_line_filter(text: str) -> bool:
    if text is None or text.startswith('[t]') or text.startswith('*') or len(text) < 10:
        return False
    else:
        return True


def get_aspects(reviews_path: str) -> pd.DataFrame:
    all_aspects = []
    for aspect_str in list(load_reviews(reviews_path).dropna('index').aspects):
        aspects = aspect_str.split(',')
        for aspect in [a for a in aspects if len(a) > 5]:
            all_aspects.append(get_sentiment_from_aspect_sentiment_text(aspect))
    return pd.DataFrame(all_aspects)


def get_aspects_for_each_line(aspects_str: str) -> pd.DataFrame:
    all_aspects = []
    aspects = aspects_str.split(',')
    for aspect in [a for a in aspects if len(a) > 5]:
        all_aspects.append(get_sentiment_from_aspect_sentiment_text(aspect))
    return pd.DataFrame(all_aspects)


def get_sentiment_from_aspect_sentiment_text(aspect_with_sentiment: str) -> AspectSentiment:
    aspect_with_sentiment = aspect_with_sentiment.strip()
    # TODO: skip implicit aspects
    aspect_with_sentiment = aspect_with_sentiment.replace('[u]', '').replace('[s]', '').replace('[p]', '')
    aspect_with_sentiment = aspect_with_sentiment.replace('[cs]', '').replace('(cs)', '').replace('[cc]', '')
    aspect_with_sentiment = aspect_with_sentiment.replace('{', '[').replace('}', ']')

    aspect_splitted = aspect_with_sentiment.split('[')
    aspect = aspect_splitted[0]
    try:
        sentiment_str = aspect_splitted[1].replace(']', '')
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
    except IndexError:
        click.echo(f'Error with: {aspect_with_sentiment}')
        sentiment = 0
        aspect = aspect_with_sentiment
    return AspectSentiment(aspect=aspect, sentiment=sentiment)


def aspect_distribution(reviews_path: str, draw: bool = False, freq_threshold=2) -> pd.Series:
    df = get_aspects(reviews_path)
    counts = df.aspect.value_counts()
    # take only aspects that appeared at least once
    if draw:
        plt.figure(figsize=(20, 8))
        plt.title(f'Aspect distribution for {basename(reviews_path)}')
        counts[counts >= freq_threshold].plot(kind='bar')
    return counts


def get_number_of_reviews(reviews_path: str) -> int:
    return load_reviews(reviews_path).shape[0]


def get_number_of_reviews_with_aspects(reviews_path: str) -> int:
    return load_reviews(reviews_path).dropna('index').shape[0]


def get_datasets_sizes(reviews_paths):
    return pd.DataFrame([
        DatasetSize(
            dataset=basename(reviews_path),
            size=get_number_of_reviews(reviews_path),
            reviews_word_average=np.average(load_reviews(reviews_path).text.apply(lambda t: len(t.split()))),
            aspect_coverage=get_number_of_reviews_with_aspects(reviews_path) / get_number_of_reviews(reviews_path)
        )
        for reviews_path in reviews_paths
    ])


def get_aspect_frequency_ranking_with_counts(reviews_path: str, top_n: int = 10) -> Dict:
    df = get_aspects(reviews_path)
    return df.aspect.value_counts().head(top_n).to_dict()


def get_aspect_frequency_ranking(reviews_path: str, top_n: int = 10) -> List:
    df = get_aspects(reviews_path)
    return list(df.aspect.value_counts().head(top_n).index)


def load_all_aspects_from_datasets() -> Dict:
    return {
        basename(review_path): get_aspects(review_path)
        for review_path
        in settings.BING_LIU_ASPECT_DATASETS_PATHS
    }


if __name__ == '__main__':
    all_datasets = load_all_aspects_from_datasets()
    # for reviews_path in reviews_paths:
    #     get_aspect_frequency_ranking(reviews_path)
    #     df = aspect_distribution(reviews_path)
    # example for jupyter
    datasets_sizes_df = get_datasets_sizes(settings.BING_LIU_ASPECT_DATASETS_PATHS)
    datasets_reviews_word_average_df = get_datasets_sizes(settings.BING_LIU_ASPECT_DATASETS_PATHS).plot(
        kind='bar', x='dataset', y='reviews_word_average')
