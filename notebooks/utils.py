from os.path import join
import cPickle


def load_serialized(f_path):
    """ loads pickled files from specified path """
    with open(join(f_path)) as f:
        return cPickle.load(f)
