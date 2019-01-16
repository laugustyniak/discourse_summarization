import collections

from more_itertools import flatten


def get_count_from_series(series):
    """
    Count occurences of each element from Data Frame series, that consists
    of lists of elements
    """
    return collections.Counter(flatten(series))
