#!/bin/bash

cd $HOME/resources
tar xvf nltk-2.0b9.tar.gz
cd $HOME/resources/nltk-2.0b9

python setup.py build
python setup.py install
