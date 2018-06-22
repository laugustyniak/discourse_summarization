import re
from collections import namedtuple
from os.path import basename
from pathlib import Path
from typing import List, Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from aspects.analysis.statistics_bing_liu import load_reviews, get_sentiment_from_aspect_sentiment_text
from aspects.utilities import settings
from aspects.utilities.common_nlp import load_spacy

TextTag = namedtuple('TextTag', 'text, tag')
TextTagged = namedtuple('TextTagged', 'text, tagged')

nlp = load_spacy()


def bing_liu_add_bio_tags(min_aspect_len: int = 5) -> Iterable[Tuple[str, str, str, List]]:
    """
    Parse lines such as

        aspect[sentiment] ## text

    into BIO format, each line is one text with added BIO-tags just after each token.
    B-Aspect, I-Aspect, O [no tag]
    """
    for review_path in settings.ALL_BING_LIU_REVIEWS_PATHS:
        review_df = load_reviews(review_path)
        for idx, aspects_str, text in tqdm(review_df.itertuples(), desc=f'Dataset: {review_path}'):
            text = text.strip().replace(' o ', '').replace(' O ', '')
            if isinstance(aspects_str, str) and len(aspects_str) > 0:
                aspects = [
                    get_sentiment_from_aspect_sentiment_text(a).aspect
                    for a
                    in aspects_str.split(',')
                    if len(a) > min_aspect_len
                ]
                aspects_replacement = _create_bio_replacement([
                    TextTag(text=aspect, tag='aspect')
                    for aspect
                    in aspects
                ])

                yield _entity_replecement(text, aspects_replacement), basename(review_path), aspects_str, aspects

            else:
                yield _add_bio_o_tag(text), basename(review_path), aspects_str, []


def _create_bio_replacement(text_tags: List[TextTag]) -> Iterable[TextTagged]:
    for text_tag in text_tags:
        if text_tag.text:
            for token_id, token in enumerate(text_tag.text.split(), start=1):
                if token_id == 1:  # begin bio tag
                    text_with_tags = f'{token} B-{text_tag.tag}'
                else:  # inside bio tags
                    text_with_tags += f' {token} I-{text_tag.tag}'
            yield TextTagged(f'{_add_bio_o_tag(text_tag.text)}', text_with_tags)
        else:
            yield TextTagged('', '')


def _add_bio_o_tag(text: str) -> str:
    return ' '.join([
        f'{token.text} O'
        for token
        in nlp(text)
        if token
    ])


def _entity_replecement(text: str, texts_tagged: Iterable[TextTagged]) -> str:
    text = _add_bio_o_tag(text)
    for text_tagged in texts_tagged:
        text = re.sub(text_tagged.text, text_tagged.tagged, text, flags=re.I)
    return text


def create_train_test_files(output_path: Path, text_col: str = 'text', test_size: float = 0.2):
    df = pd.read_csv(settings.BING_LIU_BIO_DATASET / 'merged_review_datasets_bio_tags.csv')
    df = df.sample(frac=1)  # shuffle rows
    X = df[text_col].tolist()
    X_train, X_test, _, _ = train_test_split(X, range(len(X)), test_size=test_size, random_state=42)

    with open(output_path / 'train.csv', 'w') as train_file:
        train_file.write(''.join([_new_line_every_tag(x) for x in X_train]))
    with open(output_path / 'test.csv', 'w') as test_file:
        test_file.write(''.join([_new_line_every_tag(x) for x in X_test]))


def _new_line_every_tag(sentence: str):
    return ''.join([
        f'{token}\n' if token.startswith('B-') or token.startswith('I-') or (
                token.startswith('O') and len(token) == 1) else f'{token} '
        for token
        in sentence.split()
    ]) + '\n'  # additional space to split documents in BIO conll format


if __name__ == '__main__':
    df = pd.DataFrame(bing_liu_add_bio_tags(), columns=['text', 'dataset', 'aspect_str', 'aspects'])
    settings.BING_LIU_BIO_DATASET.mkdir(exist_ok=True)
    df.to_csv(settings.BING_LIU_BIO_DATASET / 'merged_review_datasets_bio_tags.csv')
    create_train_test_files(settings.BING_LIU_BIO_DATASET)
