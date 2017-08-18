import cPickle


class Serializer(object):
    def __init__(self):
        pass

    def load(self, filepath):
        """Load serialized data"""
        with open(filepath, "rb") as fo:
            return cPickle.load(fo)

    def save(self, data, filename):
        """Save serialized data"""
        with open(filename, "wb") as fo:
            cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

    def append_serialized(self, string_data, filename):
        """Append serialized data"""
        with open(filename, "a") as fo:
            fo.write(string_data)
