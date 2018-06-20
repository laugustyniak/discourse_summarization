import pytest
from hamcrest import assert_that, equal_to

from aspects.preprocessing import transform_formats


@pytest.mark.parametrize("text, tag, bio_replacer", [
    (
            'ipod',
            'aspect',
            {
                'ipod': 'ipod B-aspect'
            }
    ),
    (
            'samsung note II',
            'aspect',
            {
                'samsung note II': 'samsung B-aspect note I-aspect II I-aspect'
            }
    ),
    (
            '',
            'aspect',
            {}
    ),
])
def test_create_bio_regex_replacer(text, tag, bio_replacer):
    bio_replacer_obtained = transform_formats._create_bio_regex_replacer(text, tag)
    assert_that(bio_replacer, equal_to(bio_replacer_obtained))
