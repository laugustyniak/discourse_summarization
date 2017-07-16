#!/bin/bash


wget http://pyyaml.org/download/pyyaml/PyYAML-3.09.tar.gz
tar -zxf PyYAML-3.09.tar.gz
cd PyYAML-3.09
python setup.py install
cd ..

wget https://pypi.python.org/packages/source/n/nltk/nltk-2.0b9.tar.gz
tar -zxf nltk-2.0b9.tar.gz
cd nltk-2.0b9
python setup.py install
cd ..

wget http://www.cs.toronto.edu/~weifeng/software/discourse_parse-2.01.tar.gz
tar -zxf discourse_parse-2.01.tar.gz
cd gCRF_dist/tools/crfsuite

wget https://github.com/downloads/chokkan/liblbfgs/liblbfgs-1.10.tar.gz
tar -zxf liblbfgs-1.10.tar.gz
cd liblbfgs-1.10
./configure --prefix=$HOME/local
make
make install
cd ..


cd crfsuite-0.12
chmod +x configure
./configure --prefix=$HOME/local --with-liblbfgs=$HOME/local
make
make install
cd ..

cp $HOME/local/bin/crfsuite crfsuite-stdin
chmod +x crfsuite-stdin
cd ../../..

cd gCRF_dist/tools/crfsuite
./crfsuite-stdin tag -pi -m ../../model/tree_build_set_CRF/label/intra.crfsuite test.txt