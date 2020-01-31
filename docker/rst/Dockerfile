FROM java:7

RUN python --version

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py

RUN apt-get update && apt-get install nginx build-essential python python-dev -y

RUN pip install numpy==1.13.3 \
    flask==1.1.1 \
    flask_restful==0.3.7 \
    pathlib==1.0.1 \
    scipy==1.2.1 \
    # otherwise with version 10.1+ we get AttributeError: 'module' object has no attribute 'finders'
    setuptools==9.1 \
    six==1.10.0 \
    spacy==1.9 \
    uwsgi==2.0.18

RUN python -m spacy download en

WORKDIR /app
COPY . /app

RUN tar -zxf PyYAML-3.09.tar.gz \
    && cd PyYAML-3.09 \
    && python setup.py install

RUN tar -zxf nltk-2.0b9.tar.gz \
	&& cd nltk-2.0b9 \
	&& python setup.py install

# gCRF and other resources
RUN cd tools/crfsuite/liblbfgs-1.10 \
    && ./configure --prefix=$HOME/local \
    && make \
    && make install

RUN cd tools/crfsuite/crfsuite-0.12 \
    && chmod +x configure \
    && ./configure --prefix=$HOME/local --with-liblbfgs=$HOME/local \
    && make \
    && make install

RUN cd tools/crfsuite \
    && cp $HOME/local/bin/crfsuite crfsuite-stdin \
    && chmod +x crfsuite-stdin

RUN ln -s /app/tools/crfsuite/crfsuite-0.12/lib/crf/.libs/libcrfsuite-0.12.so /usr/lib/libcrfsuite-0.12.so
RUN ln -s /app/tools/crfsuite/crfsuite-0.12/lib/liblbfgs-1.10.so /usr/lib/liblbfgs-1.10.so
RUN ln -s /app/tools/crfsuite/crfsuite-0.12/lib/cqdb/.libs/libcqdb-0.12.so /usr/lib/libcqdb-0.12.so

RUN cd tools/crfsuite \
    && ./crfsuite-stdin tag -pi -m ../../model/tree_build_set_CRF/label/intra.crfsuite test.txt

COPY nginx.conf /etc/nginx

RUN chmod +x start.sh
#CMD ["./start.sh"]