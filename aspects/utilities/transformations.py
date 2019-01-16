import pickle
from collections import defaultdict
from os.path import join


def load_serialized(f_path):
    """ loads pickled files from specified path """
    with open(join(f_path)) as f:
        return pickle.load(f)


def merge_dicts_by_key(*args):
    merged_dict = defaultdict(list)
    for d in args:
        for key, value in d.items():
            merged_dict[key].append(value)
    return merged_dict
