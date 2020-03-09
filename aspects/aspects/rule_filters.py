from functools import partial
from itertools import groupby
from typing import List, Callable

from aspects.rst.edu_tree_rules_extractor import EDURelation


def filter_rules_gerani(rules: List[EDURelation], aggregation_fn: Callable = None) -> List[EDURelation]:
    """
    Filter rules by its weight

    Parameters
    ----------
        :param rules: List of relation tuples
        :param aggregation_fn: Function to aggregate repeated tuple's weight. By default it is max weight.

    Returns
    -------
    rules_filtered : list
        List of rules/relations between aspects with their maximum gerani weights.

        Examples
        [
            Relation(aspect1=u'screen', aspect2=u'phone', relation_type='Elaboration', weight=1.38),
            Relation(aspect1=u'speaker', aspect2=u'sound', relation_type='Elaboration', weight=0.29)
        ]
    """
    if aggregation_fn is None:
        aggregation_fn = partial(max)

    return [
        EDURelation(
                *(group + (aggregation_fn([rel.weight for rel in relations]), ))
        )
        for group, relations
        in groupby(sorted(rules), key=lambda rel: rel[:3])
    ]

