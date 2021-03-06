from functools import partial
from statistics import mean

import pytest
from hamcrest import assert_that, equal_to

from aspects.aspects.rule_filters import filter_rules_gerani, filter_top_n_rules
from aspects.rst.edu_tree_rules_extractor import EDURelation


def _with_rules():
    return [
        EDURelation(0, 1, weight=10, relation_type="Elaboration"),
        EDURelation(0, 1, weight=20, relation_type="Elaboration"),
        EDURelation(0, 1, weight=11, relation_type="Contrast"),
        EDURelation(0, 1, weight=22, relation_type="Contrast"),
        EDURelation(2, 1, weight=10, relation_type="Elaboration"),
        EDURelation(0, 1, weight=30, relation_type="Elaboration"),
    ]


@pytest.mark.parametrize(
    "rules, aggregation_fn",
    [
        (
            [
                EDURelation(0, 1, weight=30, relation_type="Elaboration"),
                EDURelation(0, 1, weight=22, relation_type="Contrast"),
                EDURelation(2, 1, weight=10, relation_type="Elaboration"),
            ],
            partial(max),
        ),
        (
            [
                EDURelation(0, 1, weight=10, relation_type="Elaboration"),
                EDURelation(0, 1, weight=11, relation_type="Contrast"),
                EDURelation(2, 1, weight=10, relation_type="Elaboration"),
            ],
            partial(min),
        ),
    ],
)
def test_gerani_rules_filtering_max_for_repeated(rules, aggregation_fn):
    rules_filtered = filter_rules_gerani(_with_rules(), aggregation_fn)
    assert_that(set(rules_filtered), equal_to(set(rules)))


@pytest.mark.parametrize(
    "rules, aggregation_fn, top_n",
    [
        (
            [
                EDURelation(0, 1, weight=30, relation_type="Elaboration"),
                EDURelation(0, 1, weight=22, relation_type="Contrast"),
                EDURelation(2, 1, weight=10, relation_type="Elaboration"),
            ],
            partial(max),
            None,
        ),
        (
            [
                EDURelation(0, 1, weight=30, relation_type="Elaboration"),
            ],
            partial(min),
            1,
        ),
        (
            [
                EDURelation(0, 1, weight=30, relation_type="Elaboration"),
                EDURelation(0, 1, weight=20, relation_type="Elaboration"),
                EDURelation(0, 1, weight=16.5, relation_type="Contrast"),
            ],
            partial(mean),
            3,
        ),
    ],
)
def test_top_n_rules(rules, aggregation_fn, top_n):
    rules_filtered = filter_top_n_rules(_with_rules(), aggregation_fn, top_n)
    assert_that(set(rules_filtered), equal_to(set(rules)))
