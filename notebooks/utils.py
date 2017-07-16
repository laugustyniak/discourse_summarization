from os.path import join
import cPickle
import itertools


def load_serialized(f_path):
    """ loads pickled files from specified path """
    with open(join(f_path)) as f:
        return cPickle.load(f)


def flatten_list(list2d):
    """ flatten nested lists [[], []] -> [] """
    return list(itertools.chain(*list2d))
