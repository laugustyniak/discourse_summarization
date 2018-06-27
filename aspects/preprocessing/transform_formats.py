import re
from collections import namedtuple
from os.path import basename
from pathlib import Path
from typing import List, Iterable, Tuple

import click
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
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
            text = _clean_text_for_conll_format(text)
            if isinstance(aspects_str, str) and len(aspects_str) > 0:
                aspects = [
                    get_sentiment_from_aspect_sentiment_text(a).aspect
                    for a
                    in aspects_str.split(',')
                    if len(a) > min_aspect_len
                ]
                texts_tags_replacement = _create_BI_replacement([
                    TextTag(text=aspect, tag='aspect')
                    for aspect
                    in aspects
                ])

                yield replace_BI_conll_tags(text, texts_tags_replacement), basename(review_path), aspects_str, aspects

            else:
                yield _add_o_tag_after_every_word(text), basename(review_path), aspects_str, []


def _clean_text_for_conll_format(text: str):
    return text.strip().replace(' o ', '').replace(' O ', '').replace('I-', 'I').replace('(', '').replace(')', '')


def _create_BI_replacement(text_tags: Iterable[TextTag]) -> Iterable[TextTagged]:
    """ TextTag means text [word] that should be extended with specific tag, eg., TextTag(text='ipod', tag='aspect') """
    for text_tag in text_tags:
        if text_tag.text:
            for word_id, word in enumerate(word_tokenize(text_tag.text), start=1):
                if word_id == 1:  # begin bio tag
                    text_with_tags = f'{word} B-{text_tag.tag}'
                else:  # inside bio tags
                    text_with_tags += f' {word} I-{text_tag.tag}'
            yield TextTagged(f'{_add_o_tag_after_every_word(text_tag.text)}', text_with_tags)
        else:
            yield TextTagged('', '')


def _add_o_tag_after_every_word(text: str) -> str:
    return ' '.join([
        f'{token} O'
        for token
        in word_tokenize(text)
        if token
    ])


def replace_BI_conll_tags(text: str, texts_tagged: Iterable[TextTagged]) -> str:
    text = _add_o_tag_after_every_word(text)
    for text_tagged in texts_tagged:
        text = re.sub(text_tagged.text, text_tagged.tagged, text, flags=re.I)
    return text


def create_bing_liu_train_test_as_conll_files(
        output_path: Path, text_col: str = 'text', test_size: float = 0.2, dataset: str = None):
    df = pd.read_csv(settings.BING_LIU_BIO_DATASET / 'bing_liu_merged_review_datasets_bio_tags.csv')
    if dataset is not None:
        df = df[df.dataset == dataset]
    df = df.sample(frac=1)  # shuffle rows
    X = df[text_col].tolist()
    X_train, X_test, _, _ = train_test_split(X, range(len(X)), test_size=test_size, random_state=42)

    save_as_conll_file(output_path=output_path / 'train.conll', sentences=X_train)
    save_as_conll_file(output_path=output_path / 'test.conll', sentences=X_test)


def save_as_conll_file(output_path: Path, sentences: Iterable[str]):
    with open(output_path, 'w') as train_file:
        train_file.write(
            ''.join([
                _split_sentence_with_tags_into_word_tags_per_line(sentence.replace('\nO\n', '\n'))
                for sentence
                in sentences
            ]))


def _split_sentence_with_tags_into_word_tags_per_line(sentence: str):
    return '-DOCSTART- -X- O O\n\n' + ''.join([
        f'{token}\n' if token.startswith('B-') or token.startswith('I-') or token == 'O' else f'{token} '
        for token
        in sentence.split()
    ]) + '\n'  # additional space to split documents in BIO conll format


def parse_semeval_xml(xml_path: Path):
    with open(xml_path) as xml_file:
        soup = BeautifulSoup(xml_file, 'lxml')

    for sentence in tqdm(soup.findAll('sentence')):
        text = sentence.text.strip()
        aspects = [aspect['term'] for aspect in sentence.findAll('aspectterm')]
        aspects_categories = [aspect['category'] for aspect in sentence.findAll('aspectcategory')]
        yield text, aspects, aspects_categories


def validate_conll_format(conll_file_path: str, n_tags: int = 1):
    n_errors = 0
    errors = {}
    n_tokens = n_tags + 1

    conll_file_check = open(conll_file_path, 'r').readlines()

    with open(conll_file_path, 'w') as conll_file:
        for line_number, line in enumerate(conll_file_check, start=1):
            if '-DOCSTART- -X- O O' not in line:
                tokens = line.split()
                if len(tokens) != n_tokens and len(tokens):
                    click.echo(f'Error in line number {line_number}: {line}')
                    n_errors += 1
                    errors[line_number] = line
                    # don't save this line
                else:
                    conll_file.write(line)
            else:
                conll_file.write(line)
    if not n_errors:
        click.echo(f'No errors found in {conll_file_path}.')
    else:
        click.echo(f'#{n_errors} found in file {conll_file_path}, errors {errors}')

    return errors


def transform_semeval_xml_to_conll(xml_path: Path):
    conll_file_path = xml_path.with_suffix('.conll')
    df = pd.DataFrame(parse_semeval_xml(xml_path), columns=['text', 'aspects', 'aspects_categories'])
    df.text = df.text.apply(lambda text: _clean_text_for_conll_format(text))

    df['texts_tagged'] = ([
        _create_BI_replacement([
            TextTag(text=aspect.replace('(', '').replace(')', ''), tag='aspect')
            for aspect
            in aspects
        ])
        for aspects
        in tqdm(df.aspects, desc='Create tags to replace')
    ])

    texts = [
        replace_BI_conll_tags(row.text, row.texts_tagged)
        for _, row
        in tqdm(df.iterrows(), desc='Replace tags')
    ]

    save_as_conll_file(conll_file_path, texts)
    click.echo(f'{conll_file_path} has been saved!')


def prepare_and_validate_semeval_2014_data():
    transform_semeval_xml_to_conll(settings.SEMEVAL_RESTAURANTS_TRAIN_XML)
    transform_semeval_xml_to_conll(settings.SEMEVAL_RESTAURANTS_TEST_XML)
    validate_conll_format(settings.SEMEVAL_RESTAURANTS_TRAIN_XML.with_suffix('.conll'))
    validate_conll_format(settings.SEMEVAL_RESTAURANTS_TEST_XML.with_suffix('.conll'))

    transform_semeval_xml_to_conll(settings.SEMEVAL_LAPTOPS_TRAIN_XML)
    transform_semeval_xml_to_conll(settings.SEMEVAL_LAPTOPS_TEST_XML)
    validate_conll_format(settings.SEMEVAL_LAPTOPS_TRAIN_XML.with_suffix('.conll'))
    validate_conll_format(settings.SEMEVAL_LAPTOPS_TEST_XML.with_suffix('.conll'))


if __name__ == '__main__':
    prepare_and_validate_semeval_2014_data()
    #
    # df = pd.DataFrame(bing_liu_add_bio_tags(), columns=['text', 'dataset', 'aspect_str', 'aspects'])
    # settings.BING_LIU_BIO_DATASET.mkdir(exist_ok=True)
    # df.to_csv(settings.BING_LIU_BIO_DATASET / 'merged_review_datasets_bio_tags.csv')
    #
    # create_train_test_files(settings.BING_LIU_BIO_DATASET)
    # validate_bio_format(settings.BING_LIU_BIO_DATASET / 'train.csv')
    # validate_bio_format(settings.BING_LIU_BIO_DATASET / 'test.csv')
