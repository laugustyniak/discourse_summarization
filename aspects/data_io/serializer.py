import logging
import pickle

log = logging.getLogger(__name__)


def load(filepath):
    with open(filepath, "rb") as fo:
        pkl = pickle.load(fo)
        log.info('File loaded: {}'.format(filepath))
        return pkl


def save(data, filename):
    """Save serialized data"""
    with open(filename, "wb") as fo:
        log.info('File {} will be serialized.'.format(filename))
        pickle.dump(data, fo)


def append_serialized(string_data, filename):
    """Append serialized data"""
    with open(filename, "a") as fo:
        log.info('File {} will be serialized (appended).'.format(filename))
        fo.write(string_data)
