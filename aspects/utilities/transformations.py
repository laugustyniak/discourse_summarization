from os.path import join
import cPickle
import itertools
from collections import defaultdict


def load_serialized(f_path):
    """ loads pickled files from specified path """
    with open(join(f_path)) as f:
        return cPickle.load(f)


def flatten_list(list2d):
    """ flatten nested lists [[], []] -> [] """
    list2d = [x for x in list2d if type(x) != float]
    return list(itertools.chain(*list2d))


def merge_dicts_by_key(*args):
    merged_dict = defaultdict(list)
    for d in args:
        for key, value in d.iteritems():
            merged_dict[key].append(value)
    return merged_dict
