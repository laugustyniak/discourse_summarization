import json
from pathlib import Path

import click
from tqdm import tqdm

from aspects.data_io.parsers import parse_gzip
from aspects.utilities.settings import AMAZON_REVIEWS_DATASETS_PATH


@click.command()
@click.option('--dataset-path', required=False, type=str, help='Path to the amazon domain gz file')
@click.option('--column', required=False, type=str, default='reviewText')
@click.option('--n-reviews', required=False, type=int, default=None)
def amazon_domain_gz_to_json(dataset_path, column='reviewText', n_reviews: int = None):
    reviews = {}
    dataset_name = Path(dataset_path).name.replace('.gz', '')
    json_output = AMAZON_REVIEWS_DATASETS_PATH / dataset_name
    for idx, d in tqdm(enumerate(parse_gzip(dataset_path)), desc=f'JSON is saving to: {json_output.as_posix()}'):
        reviews[idx] = d[column]
        if n_reviews is not None and idx > n_reviews:
            break
    with open(json_output.as_posix(), 'w') as j:
        json.dump(reviews, j)


if __name__ == '__main__':
    amazon_domain_gz_to_json()
