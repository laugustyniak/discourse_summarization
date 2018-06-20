import pytest
from hamcrest import assert_that, equal_to

from aspects.preprocessing.transform_formats import TextTag, TextTagged, _create_bio_replacement


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
