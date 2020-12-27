from functools import partial
from itertools import groupby
from operator import attrgetter
from typing import Callable, List

from aspects.rst.edu_tree_rules_extractor import EDURelation


def filter_rules_gerani(
    rules: List[EDURelation], aggregation_fn: Callable = None
) -> List[EDURelation]:
    """
    Filter rules by its weight for each Discourse Tree.
    We can have many rules per tree.

    Parameters
    ----------
        :param rules: List of relation tuples
        :param aggregation_fn: Function to aggregate repeated tuple's weight.
        By default it is max weight.

    Returns
    -------
    rules_filtered : list
        List of rules/relations between aspects with their maximum
        gerani weights.

        Examples
        [
            EDURelation(edu1=u'screen', edu2=u'phone',
                relation_type='Elaboration', weight=1.38),
            EDURelation(edu1=u'speaker', edu2=u'sound',
                relation_type='Elaboration', weight=0.29),
            EDURelation(edu1=u'speaker', edu2=u'sound',
                relation_type='Elaboration', weight=0.21)
        ]
    """
    if aggregation_fn is None:
        aggregation_fn = partial(max)

    rules = sorted(rules, key=lambda relation: relation[:3])
    return [
        EDURelation(*(group + (aggregation_fn([r.weight for r in relations]),)))
        for group, relations in groupby(
            rules, key=lambda relation: relation[:3]
        )
    ]


def filter_top_n_rules(
    rules: List[EDURelation], aggregation_fn: Callable = None, top_n: int = 1
) -> List[EDURelation]:
    if aggregation_fn is None:
        aggregation_fn = partial(max)

    # sort for groupby
    rules = sorted(rules, key=lambda relation: relation[:3])

    rules_filtered = [
        EDURelation(*(group + (aggregation_fn([r.weight for r in relations]),)))
        for group, relations in groupby(rules, key=lambda relation: relation[:3])
    ]

    if top_n is None:
        return rules_filtered
    else:
        return sorted(rules, key=attrgetter("weight"), reverse=True)[:top_n]
