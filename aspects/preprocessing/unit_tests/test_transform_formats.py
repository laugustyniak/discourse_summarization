import pytest
from hamcrest import assert_that, equal_to

from aspects.preprocessing.transform_formats import (
    TextTag, TextTagged, _create_bio_replacement, _add_bio_o_tag, _entity_replecement, _new_line_every_tag)


@pytest.mark.parametrize("text_tag, bio_replacer", [
    (
            [TextTag('ipod', 'aspect')],
            [TextTagged('ipod O-tag', 'ipod\tB-aspect')]
    ),
    (
            [TextTag('samsung note II', 'aspect')],
            [TextTagged('samsung O-tag note O-tag II O-tag', 'samsung\tB-aspect\nnote\tI-aspect\nII\tI-aspect')]
    ),
    (
            [TextTag('', 'aspect')],
            [TextTagged('', '')]
    ),
])
def test_create_bio_regex_replacer(text_tag, bio_replacer):
    bio_replacer_obtained = list(_create_bio_replacement(text_tag))
    assert_that(bio_replacer_obtained, equal_to(bio_replacer))


@pytest.mark.parametrize("text, text_tagged", [
    (
            'samsung note II',
            'samsung O-tag note O-tag II O-tag'
    ),
    (
            'I like it iphone',
            'I O-tag like O-tag it O-tag iphone O-tag'
    ),
])
def test_add_bio_o_tag(text, text_tagged):
    bio_replacer_obtained = _add_bio_o_tag(text)
    assert_that(bio_replacer_obtained, equal_to(text_tagged))


@pytest.mark.parametrize("text, texts_tagged, text_expected", [
    (
            'I like samsung note II',
            [
                TextTagged(
                    'samsung O-tag note O-tag II O-tag',
                    'samsung B-aspect note I-aspect II I-aspect'
                )
            ],
            'I O-tag like O-tag samsung B-aspect note I-aspect II I-aspect',

    ),
    (
            'I like this iphone',
            [
                TextTagged(
                    'iphone O-tag',
                    'iphone B-product'
                )
            ],
            'I O-tag like O-tag this O-tag iphone B-product'
    ),
])
def test_entity_replecement(text, texts_tagged, text_expected):
    text_obtained = _entity_replecement(text, texts_tagged)
    assert_that(text_obtained, equal_to(text_expected))


@pytest.mark.parametrize("text, text_output", [
    (
            'I O-tag like O-tag samsung B-aspect note I-aspect II I-aspect',
            'I\tO-tag\nlike\tO-tag\nsamsung\tB-aspect\nnote\tI-aspect\nII\tI-aspect\n\n'
    )
])
def test_new_line_every_tag(text, text_output):
    text_obtained = _new_line_every_tag(text)
    assert_that(text_obtained, equal_to(text_output))
