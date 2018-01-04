class WrongTypeException(Exception):
    def __init__(self, extension):
        Exception.__init__(self,
                           'Wrong file type -> extension: {}'.format(extension))
