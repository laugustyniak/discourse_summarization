from collections import namedtuple
from typing import List

from aspects.analysis.statistics_bing_liu import load_reviews, get_sentiment_from_aspect_sentiment_text
from aspects.utilities import common_nlp
from aspects.utilities import settings

TextTag = namedtuple('TextTag', 'text, tag')

nlp = common_nlp.load_spacy()


def bing_liu_2_conll_format():
    """
    Parse lines such as

        aspect[sentiment] ## text

    into BIO format, each line is one text with added BIO-tags just after each token.
    B-Aspect, I-Aspect, O [no tag]
    """
    for review_path in settings.ALL_BING_LIU_REVIEWS_PATHS:
        review_df = load_reviews(review_path)
        for idx, aspects_str, text in review_df.itertuples():
            if isinstance(aspects_str, str):
                aspects = [
                    get_sentiment_from_aspect_sentiment_text(a).aspect
                    for a
                    in aspects_str.split()
                ]

                for aspect_str in aspects:
                    for aspect in aspects_str.split():
                        pass

            else:
                aspects = []

    return review_df


def extend_text_with_bio_tags(text: str, tags: List[TextTag]):
    tokens_with_bio_tags = []
    for token in nlp(text):
        tokens_with_bio_tags.append(token)
        tokens_with_bio_tags.append(tag_token(token, [], tokens_with_bio_tags[-1]))

    return


def _create_bio_regex_replacer(text: str, tag: str):
    if not text:
        return {}

    for token_id, token in enumerate(nlp(text), start=1):
        if token_id == 1:  # begin bio tag
            text_with_tags = f'{token.text} B-{tag}'
        else:  # inside bio tags
            text_with_tags += f' {token.text} I-{tag}'
    return {
        text: text_with_tags
    }


def tag_token(token: str, tags, prev_tag):
    return


if __name__ == '__main__':
    outcome = bing_liu_2_conll_format()
    pass
