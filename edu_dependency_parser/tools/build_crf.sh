gCRF_ROOT=$HOME/sa/edu_dependency_parser

cd $gCRF_ROOT/tools/crfsuite/liblbfgs-1.10
./configure --prefix=$HOME/local
make
make install

cd $gCRF_ROOT/tools/crfsuite/crfsuite-0.12
./configure --prefix=$HOME/local --with-liblbfgs=$HOME/local
make
make install

cp $HOME/local/bin/crfsuite $gCRF_ROOT/tools/crfsuite/crfsuite-stdin
chmod +x ../crfsuite-stdin
