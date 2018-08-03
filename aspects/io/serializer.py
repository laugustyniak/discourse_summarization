import logging
import pickle

log = logging.getLogger(__name__)


class Serializer(object):

    def load(self, filepath):
        """Load serialized data"""
        try:
            with open(filepath, "rb") as fo:
                pkl = pickle.load(fo)
                log.info('File loaded: {}'.format(filepath))
                return pkl
        except IOError as err:
            log.error('Error {}'.format(str(err)))

    def save(self, data, filename):
        """Save serialized data"""
        with open(filename, "wb") as fo:
            log.info('File {} will be serialized.'.format(filename))
            pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)

    def append_serialized(self, string_data, filename):
        """Append serialized data"""
        with open(filename, "a") as fo:
            log.info('File {} will be serialized (appended).'.format(filename))
            fo.write(string_data)
