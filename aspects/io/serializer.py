import cPickle
import logging

log = logging.getLogger(__name__)


class Serializer(object):
    def __init__(self):
        pass

    def load(self, filepath):
        """Load serialized data"""
        try:
            with open(filepath, "rb") as fo:
                pkl = cPickle.load(fo)
                log.info('File loaded: {}'.format(filepath))
                return pkl
        except IOError as err:
            log.error('Error {}'.format(str(err)))

    def save(self, data, filename):
        """Save serialized data"""
        with open(filename, "wb") as fo:
            log.info('File {} will be serialized.'.format(filename))
            cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

    def append_serialized(self, string_data, filename):
        """Append serialized data"""
        with open(filename, "a") as fo:
            log.info('File {} will be serialized (appended).'.format(filename))
            fo.write(string_data)
