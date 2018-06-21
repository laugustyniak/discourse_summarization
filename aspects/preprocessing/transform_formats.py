import csv
import re
from collections import namedtuple
from typing import List, Iterable

import click
from tqdm import tqdm

from aspects.analysis.statistics_bing_liu import load_reviews, get_sentiment_from_aspect_sentiment_text
from aspects.utilities import common_nlp
from aspects.utilities import settings

TextTag = namedtuple('TextTag', 'text, tag')
TextTagged = namedtuple('TextTagged', 'text, tagged')

nlp = common_nlp.load_spacy()


def bing_liu_2_conll_format(min_aspect_len: int = 5) -> Iterable[str]:
    """
    Parse lines such as

        aspect[sentiment] ## text

    into BIO format, each line is one text with added BIO-tags just after each token.
    B-Aspect, I-Aspect, O [no tag]
    """
    n_errors = 0
    n_total = 0
    for review_path in settings.ALL_BING_LIU_REVIEWS_PATHS:
        review_df = load_reviews(review_path)
        for idx, aspects_str, text in tqdm(review_df.itertuples()):
            n_total += 1
            if isinstance(aspects_str, str) and len(aspects_str) > 0:
                try:
                    aspects = [
                        get_sentiment_from_aspect_sentiment_text(a).aspect
                        for a
                        in aspects_str.split(',')
                        if len(a) > min_aspect_len
                    ]
                except:
                    click.echo(f'Error in {text} with aspects: {aspects_str}')
                    n_errors += 1
                    pass

                aspects_replacement = _create_bio_replacement([
                    TextTag(text=aspect, tag='aspect')
                    for aspect
                    in aspects
                ])

                yield _entity_replecement(text, aspects_replacement)

            else:
                yield _add_bio_o_tag(text)
    click.echo(f'All reviews: {n_total} and {n_errors} excepton during parsing')


def _create_bio_replacement(text_tags: List[TextTag]) -> Iterable[TextTagged]:
    for text_tag in text_tags:
        if text_tag.text:
            for token_id, token in enumerate(nlp(text_tag.text), start=1):
                if token_id == 1:  # begin bio tag
                    text_with_tags = f'{token.text} B-{text_tag.tag}'
                else:  # inside bio tags
                    text_with_tags += f' {token.text} I-{text_tag.tag}'
            yield TextTagged(f'{_add_bio_o_tag(text_tag.text)}', text_with_tags)
        else:
            yield TextTagged('', '')


def _add_bio_o_tag(text: str) -> str:
    return ' '.join([
        f'{token.text} O-tag'
        for token
        in nlp(text)
    ])


def _entity_replecement(text: str, texts_tagged: Iterable[TextTagged]) -> str:
    text = _add_bio_o_tag(text)
    for text_tagged in texts_tagged:
        text = re.sub(text_tagged.text, text_tagged.tagged, text, flags=re.I)
    return text


if __name__ == '__main__':
    outcome = list(bing_liu_2_conll_format())

    settings.BING_LIU_BIO_DATASET.mkdir(exist_ok=True)
    with open(settings.BING_LIU_BIO_DATASET / 'merged_review_datasets_bio_tags.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(outcome)
