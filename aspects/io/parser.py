import gzip

import pandas as pd
import pathlib

ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent

# reviews_Amazon_Instant_Video.json.gz
AMAZON_REVIEWS_DATASET = 'reviews_Apps_for_Android.json.gz'


def parse(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield eval(l)


def get_amazon_dataset(f_name):
    dataset_path = ROOT_PATH / 'data' / 'reviews' / 'amazon' / f_name
    i = 0
    df = {}
    for d in parse(str(dataset_path)):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df = get_amazon_dataset(AMAZON_REVIEWS_DATASET)
pass
