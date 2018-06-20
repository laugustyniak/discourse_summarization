import pytest
from hamcrest import assert_that, equal_to

from aspects.preprocessing.transform_formats import TextTag, TextTagged, _create_bio_replacement, _add_bio_o_tag


@pytest.mark.parametrize("text_tag, bio_replacer", [
    (
            [TextTag('ipod', 'aspect')],
            [TextTagged('ipod', 'ipod B-aspect')]
    ),
    (
            [TextTag('samsung note II', 'aspect')],
            [TextTagged('samsung note II', 'samsung B-aspect note I-aspect II I-aspect')]
    ),
    (
            [TextTag('', 'aspect')],
            [TextTagged('', '')]
    ),
])
def test_create_bio_regex_replacer(text_tag, bio_replacer):
    bio_replacer_obtained = list(_create_bio_replacement(text_tag))
    assert_that(bio_replacer, equal_to(bio_replacer_obtained))


@pytest.mark.parametrize("text_tag, text_tagged", [
    (
            TextTag('samsung note II', 'aspect'),
            'samsung O-aspect note O-aspect II O-aspect'
    ),
    (
            TextTag('I like it iphone', 'aspect'),
            'I O-aspect like O-aspect it O-aspect iphone O-aspect'
    ),
])
def test_add_bio_o_tag(text_tag, text_tagged):
    bio_replacer_obtained = _add_bio_o_tag(text_tag)
    assert_that(text_tagged, equal_to(bio_replacer_obtained))
