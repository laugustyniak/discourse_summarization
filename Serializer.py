# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import sys
import cPickle
import json
from pprint import pprint
import os.path


class Serializer():
    def load(self, filepath):
        try:
            fo = open(filepath, "rb")
        except IOError:
            print "Couldn't open data file for read: %s" % filepath
            return

        try:
            object = cPickle.load(fo)
        except:
            fo.close()
            print "Unexpected error:", sys.exc_info()[0]
            raise

        fo.close()

        return object

    def save(self, data, filename):
        # zapisujemy  dane zserializowane
        try:
            with open(filename, "wb") as f_p:
                cPickle.dump(data, f_p, protocol=cPickle.HIGHEST_PROTOCOL)
        except IOError:
            print "Couldn't open data file for write: %s" % filename
            return

    def append(self, stringData, filename):
        # zapisujemy dane do odczytu
        try:
            with open(filename, "a") as f_o:
                f_o.write(stringData)
        except IOError:
            print "Couldn't open data file for write: %s" % filename
            return
