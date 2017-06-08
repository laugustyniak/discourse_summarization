# -*- coding: utf-8 -*-
#author: Krzysztof xaru Rajda

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
        #zapisujemy  dane zserializowane
        try:
            f_p = open(filename, "wb")
        except IOError:
            print "Couldn't open data file for write: %s" % filename
            return
        
        cPickle.dump(data, f_p, protocol = cPickle.HIGHEST_PROTOCOL)
        f_p.close()
        
    def append(self, stringData, filename):
     #zapisujemy dane do odczytu
        try:
            f_o = open(filename, "a")
        except IOError:
            print "Couldn't open data file for write: %s" % filename
            return
        
        f_o.write(stringData)
        f_o.close()
        
       