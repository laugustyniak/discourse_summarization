FROM java:7

MAINTAINER Łukasz Augustyniak "luk.augustyniak@gmail.com"

WORKDIR /app
COPY . /app

RUN python --version

RUN apt-get update && apt-get install build-essential python python-dev -y

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py

RUN tar -zxf PyYAML-3.09.tar.gz \
    && cd PyYAML-3.09 \
    && python setup.py install

RUN tar -zxf nltk-2.0b9.tar.gz \
	&& cd nltk-2.0b9 \
	&& python setup.py install

# gCRF and other resources
RUN tar -zxf rhetorical_parser.tar.gz \
    && cd rhetorical_parser/tools/crfsuite/liblbfgs-1.10 \
    && ./configure --prefix=$HOME/local \
    && make \
    && make install

# add rhetorical source code to parser directory
#RUN mv src rhetorical_parser
#RUN ls -l rhetorical_parser/src

RUN cd rhetorical_parser/tools/crfsuite/crfsuite-0.12 \
    && chmod +x configure \
    && ./configure --prefix=$HOME/local --with-liblbfgs=$HOME/local \
    && make \
    && make install

RUN cd rhetorical_parser/tools/crfsuite \
    && cp $HOME/local/bin/crfsuite crfsuite-stdin \
    && chmod +x crfsuite-stdin

RUN cd rhetorical_parser/tools/crfsuite \
    && ./crfsuite-stdin tag -pi -m ../../model/tree_build_set_CRF/label/intra.crfsuite test.txt

RUN pip install numpy==1.13.3 \
    flask==1.1.1 \
    flask_restful==0.3.7 \
    pathlib==1.0.1 \
    scipy==1.2.1 \
    spacy==1.9

RUN python -m spacy download en

#ENTRYPOINT ["python"]
#
#CMD ["/app/rhetorical_parser/src/app.py"]