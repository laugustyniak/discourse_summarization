import gzip

import pandas as pd
import pathlib

# TODO: move it to settings
ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent

# TODO: change into ENUM
# reviews_Amazon_Instant_Video.json.gz
AMAZON_REVIEWS_DATASET = 'reviews_Apps_for_Android.json.gz'


def parse_reviews(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield eval(l)


def get_amazon_dataset(f_name):
    # TODO: move it to settings
    dataset_path = ROOT_PATH / 'data' / 'reviews' / 'amazon' / f_name
    i = 0
    reviews = {}
    for d in parse_reviews(str(dataset_path)):
        reviews[i] = d
        i += 1
    return pd.DataFrame.from_dict(reviews, orient='index')


df = get_amazon_dataset(AMAZON_REVIEWS_DATASET)
