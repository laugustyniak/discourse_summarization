import gzip
import json
from tqdm import tqdm

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


def get_amazon_dataset(f_name, column='reviewText', n_reviews=10000):
    # TODO: move it to settings
    dataset_path = str(ROOT_PATH / 'data' / 'reviews' / 'amazon' / f_name)
    i = 0
    reviews = {}
    for d in tqdm(parse_reviews(dataset_path)):
        reviews[i] = d[column]
        i += 1
        if i > n_reviews:
            break
    with open(dataset_path.replace('.gz', '')) as j:
        json.dump(reviews, j)


if __name__ == '__main__':
    get_amazon_dataset(AMAZON_REVIEWS_DATASET)
