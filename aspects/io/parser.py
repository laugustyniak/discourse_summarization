import gzip
import json

from tqdm import tqdm

from aspects.utilities import settings


def parse_reviews(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield eval(l)


def get_amazon_dataset(dataset_path, column='reviewText', n_reviews=10000):
    i = 0
    reviews = {}
    for d in tqdm(parse_reviews(dataset_path)):
        reviews[i] = d[column]
        i += 1
        if i > n_reviews:
            break
    with open(dataset_path.replace('.gz', ''), 'wb') as j:
        json.dump(reviews, j)


if __name__ == '__main__':
    get_amazon_dataset(settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_GZ)
